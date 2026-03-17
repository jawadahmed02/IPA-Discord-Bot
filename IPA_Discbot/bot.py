import os
import sqlite3
import llm
import asyncio
import threading
import json

from datetime import datetime, UTC

import discord
from discord.ext import commands
from discord import app_commands

from dotenv import load_dotenv

from cryptography.fernet import Fernet, InvalidToken

from mcp_client import solve_pddl

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BOT_MASTER_KEY = os.getenv("BOT_MASTER_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")

intents = discord.Intents.default()  # default discord events enabled
intents.message_content = True  # conversation logging

bot = commands.Bot(command_prefix="!", intents=intents)

DB_PATH = "bot.db"  # database path


# new encryption features
# Fernet - AES encryption + HMAC authentication
def _get_fernet() -> Fernet:
    if not BOT_MASTER_KEY:
        raise RuntimeError("Missing BOT_MASTER_KEY in environment")
    try:
        return Fernet(BOT_MASTER_KEY.strip().encode())
    except Exception as e:
        raise RuntimeError("BOT_MASTER_KEY is invalid") from e


def encrypt_api_key(api_key: str) -> bytes:
    return _get_fernet().encrypt(api_key.encode("utf-8"))


def decrypt_api_key(token: bytes) -> str:
    try:
        return _get_fernet().decrypt(token).decode("utf-8")
    except InvalidToken as e:
        raise RuntimeError("Failed to decrypt API key: invalid master key or corrupted data") from e

def migrate_plaintext_keys_if_needed():
    con = _db_connect()
    try:
        cur = con.cursor()

        cur.execute("PRAGMA table_info(user_provider_keys)")
        columns = cur.fetchall()
        column_names = {col[1] for col in columns}

        if "api_key" in column_names:
            print("[DB] Migrating plaintext API keys to encrypted storage...")

            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_provider_keys_new (
                    user_id TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    api_key_encrypted BLOB NOT NULL,
                    PRIMARY KEY (user_id, provider)
                )
            """)

            cur.execute("SELECT user_id, provider, api_key FROM user_provider_keys")
            rows = cur.fetchall()

            for user_id, provider, plaintext_key in rows:
                if plaintext_key is None:
                    continue
                plaintext_key = str(plaintext_key).strip()
                if not plaintext_key:
                    continue

                encrypted = encrypt_api_key(plaintext_key)
                cur.execute("""
                    INSERT OR REPLACE INTO user_provider_keys_new (user_id, provider, api_key_encrypted)
                    VALUES (?, ?, ?)
                """, (user_id, provider, encrypted))

            cur.execute("DROP TABLE user_provider_keys")
            cur.execute("ALTER TABLE user_provider_keys_new RENAME TO user_provider_keys")

            con.commit()
            print("[DB] Migration complete.")
        else:
            con.commit()
    finally:
        con.close()


def _db_connect():
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.execute("PRAGMA journal_mode=WAL;") # Improvement for better reader and writer concurrency
    con.execute("PRAGMA synchronous=NORMAL;") # Controls how aggressively SQLite syncs to disk
    con.execute("PRAGMA foreign_keys=ON;")
    return con

def init_db():
    con = _db_connect()
    cur = con.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            guild_id TEXT,
            channel_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS user_provider_keys (
            user_id TEXT NOT NULL,
            provider TEXT NOT NULL,
            api_key_encrypted BLOB NOT NULL,
            PRIMARY KEY (user_id, provider)
        );

        CREATE TABLE IF NOT EXISTS user_model_selection (
            user_id TEXT NOT NULL PRIMARY KEY,
            model_id TEXT NOT NULL
        );
    """)

    con.commit()
    con.close()

    # Run after base init so older DBs get upgraded
    migrate_plaintext_keys_if_needed()



def log_message(channel_id: int, user_id: int, role: str, content: str, guild_id: int | None):  # conv. logging for specific user
    con = _db_connect()
    cur = con.cursor()
    cur.execute(
        "INSERT INTO messages (ts, guild_id, channel_id, user_id, role, content) VALUES (?, ?, ?, ?, ?, ?)",
        (datetime.now(UTC).isoformat(), str(guild_id) if guild_id else None, str(channel_id), str(user_id), role,
         content)
    )
    con.commit()
    con.close()


def get_recent_context(channel_id: int, user_id: int, limit: int = 12):  # conv. context from channel and user
    con = _db_connect()
    cur = con.cursor()
    cur.execute(
        """
        SELECT role, content
        FROM messages
        WHERE channel_id = ?
          AND user_id = ?
        ORDER BY id DESC LIMIT ?
        """,
        (str(channel_id), str(user_id), limit)
    )
    rows = cur.fetchall()
    con.close()
    rows.reverse()  # (oldest => newest) instead of (newest => oldest) for chronological order
    return [{"role": r, "content": c} for (r, c) in rows]

# helper function for building transcript from msgs
def _build_transcript(context_messages: list[dict]) -> str:
    lines: list[str] = []
    for m in context_messages:
        role = m.get("role", "").strip().lower()
        content = (m.get("content") or "").strip()

        if role == "assistant":
            prefix = "Assistant"
        elif role == "system":
            prefix = "System"
        else:
            prefix = "User"

        lines.append(f"{prefix}: {content}")

    lines.append("Assistant:")

    return "\n".join(lines)


def get_user_model(user_id: str) -> str | None:
    con = _db_connect()
    cur = con.cursor()
    cur.execute("SELECT model_id FROM user_model_selection WHERE user_id = ?", (user_id,))
    row = cur.fetchone()
    con.close()
    return row[0] if row else None


def set_user_model(user_id: str, model_id: str):
    con = _db_connect()
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO user_model_selection (user_id, model_id)
        VALUES (?, ?) ON CONFLICT(user_id) DO
        UPDATE SET model_id=excluded.model_id
        """,
        (user_id, model_id),
    )
    con.commit()
    con.close()


def _llm_reply_sync(model_id: str, context_messages: list[dict]) -> str:
    system_prompt = (
        "You are a helpful planning assistant inside Discord. "
        "Keep answers concise."
    )

    transcript = _build_transcript(context_messages)
    model = llm.get_model(model_id)
    response = model.prompt(transcript, system=system_prompt)

    return response.text().strip()

# -------------------------------------- Model change features start here ----------------------------------------------

LLM_ENV_LOCK = threading.Lock()

PROVIDER_ENV = {
    "openai": "OPENAI_API_KEY",
    "gemini": "LLM_GEMINI_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "ollama": None,
}


def save_provider_key(user_id: str, provider: str, api_key: str):
    encrypted_key = encrypt_api_key(api_key.strip())

    con = _db_connect()
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO user_provider_keys (user_id, provider, api_key_encrypted)
        VALUES (?, ?, ?)
        ON CONFLICT(user_id, provider) DO UPDATE SET api_key_encrypted=excluded.api_key_encrypted
        """,
        (user_id, provider, encrypted_key),
    )
    con.commit()
    con.close()


def get_provider_key(user_id: str, provider: str) -> str | None:
    con = _db_connect()
    cur = con.cursor()
    cur.execute(
        "SELECT api_key_encrypted FROM user_provider_keys WHERE user_id = ? AND provider = ?",
        (user_id, provider),
    )
    row = cur.fetchone()
    con.close()

    if not row:
        return None

    encrypted_key = row[0]
    return decrypt_api_key(encrypted_key)


def _provider_from_model_id(model_id: str) -> str | None:
    mid = model_id.lower()
    if mid.startswith("ollama:") or mid.startswith("ollama/"):
        return "ollama"
    if "claude" in mid or mid.startswith("anthropic"):
        return "anthropic"
    if "gemini" in mid:
        return "gemini"
    return "openai"


def _run_llm_for_user_sync(user_id: str, model_id: str, context_messages: list[dict]) -> str:
    provider = _provider_from_model_id(model_id)
    env_key = PROVIDER_ENV.get(provider or "")

    transcript = _build_transcript(context_messages)
    system_prompt = (
        "You are a helpful planning assistant inside Discord. "
        "Keep answers concise."
    )

    # Pull the per-user key if provider needs it
    user_key = None
    if env_key:
        user_key = get_provider_key(user_id, provider)
        if not user_key:
            raise RuntimeError(f"No API key set for {provider}. Use /setkey {provider} <key>.")

    # Important: env vars are global, lock to avoid cross-user collisions
    with LLM_ENV_LOCK:
        old = os.environ.get(env_key) if env_key else None
        try:
            if env_key and user_key:
                os.environ[env_key] = user_key

            model = llm.get_model(model_id)

            resp = model.prompt(transcript, system=system_prompt)
            return resp.text().strip()




        finally:
            if env_key:
                if old is None:
                    os.environ.pop(env_key, None)
                else:
                    os.environ[env_key] = old


async def llm_reply(model_id: str, context_messages: list[dict],
                    user_id: str | None = None) -> str:  # responsible for LLM response

    print("========== MODEL DEBUG ==========")
    print("User ID:", user_id)
    print("DB model:", get_user_model(user_id) if user_id else None)
    print("Passed model_id:", model_id)
    print("Fallback MODEL constant:", MODEL)
    print("=================================")

    if user_id is None:
        return await asyncio.to_thread(_llm_reply_sync, model_id, context_messages)
    return await asyncio.to_thread(_run_llm_for_user_sync, user_id, model_id, context_messages)


def _all_llm_model_ids() -> list[str]:
    return sorted([m.model_id for m in llm.get_models()])


@bot.tree.command(name="models", description="List available models")
async def models_cmd(interaction: discord.Interaction):
    ids = _all_llm_model_ids()
    if not ids:
        await interaction.response.send_message("No models found. Install an llm provider plugin.", ephemeral=True)
        return

    # keep under Discord limits
    text = "Available models:\n" + "\n".join(f"- {mid}" for mid in ids)
    await interaction.response.send_message(text[:1900], ephemeral=True)


async def model_autocomplete(interaction: discord.Interaction, current: str,) -> list[app_commands.Choice[str]]:
    ids = _all_llm_model_ids()
    current_lower = (current or "").lower()

    matches = [mid for mid in ids if current_lower in mid.lower()]
    matches = matches[:25] # apparent discord cap, crashes otherwise

    return [app_commands.Choice(name=mid, value=mid) for mid in matches]


@bot.tree.command(name="use", description="Choose the model this bot will use for you")
@app_commands.describe(model_id="Pick from /models")
@app_commands.autocomplete(model_id=model_autocomplete)
async def use_cmd(interaction: discord.Interaction, model_id: str):
    ids = set(_all_llm_model_ids())
    if model_id not in ids:
        await interaction.response.send_message("Unknown model_id. Use /models.", ephemeral=True)
        return

    set_user_model(str(interaction.user.id), model_id)
    await interaction.response.send_message(f"Now using: {model_id}", ephemeral=True)


@bot.tree.command(name="setkey", description="Set your API key for a provider")
@app_commands.describe(provider="openai | gemini | anthropic", api_key="Your API key")
async def setkey_cmd(interaction: discord.Interaction, provider: str, api_key: str):
    provider = provider.strip().lower()
    if provider not in ("openai", "gemini", "anthropic"):
        await interaction.response.send_message("Unknown provider. Use: openai | gemini | anthropic", ephemeral=True)
        return

    save_provider_key(
        user_id=str(interaction.user.id),
        provider=provider,
        api_key=api_key.strip()
    )
    await interaction.response.send_message(
        f"Saved key for {provider}.",
        ephemeral=True
    )


# Keeps replies under Discord's message size limit
def _truncate_discord_message(text: str, limit: int = 1900) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


# Pretty-prints JSON planner output while still handling plain-text errors cleanly
def _pretty_json_or_text(text: str) -> str:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text.strip()

    return json.dumps(parsed, indent=2)


# Reads an uploaded file as UTF-8 so PDDL attachments can be sent to the planner
async def _read_text_attachment(attachment: discord.Attachment) -> str:
    data = await attachment.read()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as e:
        raise RuntimeError(f"{attachment.filename} is not valid UTF-8 text.") from e


# Pulls out the domain/problem pair expected by the default PaaS solve tool
async def _extract_pddl_attachments(message: discord.Message) -> tuple[str, str]:
    attachments = list(message.attachments)
    if len(attachments) < 2:
        raise RuntimeError("Attach both a domain PDDL file and a problem PDDL file.")

    domain_text: str | None = None
    problem_text: str | None = None

    for attachment in attachments:
        filename = attachment.filename.lower()
        content = await _read_text_attachment(attachment)

        if domain_text is None and "domain" in filename:
            domain_text = content
            continue

        if problem_text is None and "problem" in filename:
            problem_text = content

    if domain_text is None or problem_text is None:
        pddl_files = [
            attachment for attachment in attachments
            if attachment.filename.lower().endswith((".pddl", ".txt"))
        ]
        if len(pddl_files) >= 2:
            if domain_text is None:
                domain_text = await _read_text_attachment(pddl_files[0])
            if problem_text is None:
                problem_text = await _read_text_attachment(pddl_files[1])

    if domain_text is None or problem_text is None:
        raise RuntimeError(
            "Could not identify both files. Name them with `domain` and `problem`, "
            "or attach exactly two PDDL text files."
        )

    return domain_text, problem_text


# -------------------------------------- Model change features end  here -----------------------------------------------

GUILD_ID = 1376609949114699886 # uBots server id

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user} (id={bot.user.id})")
    guild = discord.Object(id=GUILD_ID)
    bot.tree.copy_global_to(guild=guild)
    synced = await bot.tree.sync(guild=guild)
    print(f"Synced {len(synced)} commands to guild {GUILD_ID}.")



@bot.command()  # Primary conversation comman
async def chat(ctx: commands.Context, *, text: str):
    # save user message
    log_message(ctx.channel.id, ctx.author.id, "user", text, ctx.guild.id if ctx.guild else None)

    async with ctx.typing():
        context = get_recent_context(ctx.channel.id, ctx.author.id, limit=12)
        selected_model = get_user_model(str(ctx.author.id)) or MODEL
        try:
            answer = await llm_reply(selected_model, context, user_id=str(ctx.author.id))
        except Exception as e:
            print("[LLM ERROR]", type(e).__name__)
            answer = "Error generating response"

    # save assistant message
    log_message(ctx.channel.id, ctx.author.id, "assistant", answer, ctx.guild.id if ctx.guild else None)

    # discord message limit safeguard
    await ctx.reply(_truncate_discord_message(answer))


# Minimal planning entrypoint: read attached PDDL files and run the default PaaS solver
@bot.command()
async def solve(ctx: commands.Context):
    async with ctx.typing():
        try:
            domain_text, problem_text = await _extract_pddl_attachments(ctx.message)
            result = await solve_pddl(domain_text, problem_text)
        except Exception as e:
            await ctx.reply(_truncate_discord_message(f"Solve failed: {e}"))
            return

    formatted = _pretty_json_or_text(result)
    await ctx.reply(_truncate_discord_message(f"```json\n{formatted}\n```"))


if __name__ == "__main__":
    if not DISCORD_TOKEN:
        raise RuntimeError("Missing DISCORD_TOKEN")
    if not BOT_MASTER_KEY:
        raise RuntimeError("Missing BOT_MASTER_KEY")

    init_db()
    bot.run(DISCORD_TOKEN)
