import os
import sqlite3
import llm
import asyncio
import threading

from datetime import datetime, UTC

import discord
from discord.ext import commands
from discord import app_commands

from dotenv import load_dotenv

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")

intents = discord.Intents.default()  # default discord events enabled
intents.message_content = True  # conversation logging

bot = commands.Bot(command_prefix="!", intents=intents)

DB_PATH = "bot.db"  # database path


# table parameters below are as listed:
# unique id
# timestamp
# conv server
# conv channel
# conv user
# conv role
# msg content

def _db_connect():
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.execute("PRAGMA journal_mode=WAL;") # Readers and writers can’t operate simultaneously
    con.execute("PRAGMA synchronous=NORMAL;") # Controls how aggressively SQLite syncs to disk
    con.execute("PRAGMA foreign_keys=ON;")
    return con

def init_db():
    con = _db_connect()  # open / create database
    cur = con.cursor()              # cursor for executing SQL

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
            api_key TEXT NOT NULL,
            PRIMARY KEY (user_id, provider)
        );
    
        CREATE TABLE IF NOT EXISTS user_model_selection (
            user_id TEXT NOT NULL PRIMARY KEY,
            model_id TEXT NOT NULL
        );
    """)

    con.commit()
    con.close()



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
    con = _db_connect()
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO user_provider_keys (user_id, provider, api_key)
        VALUES (?, ?, ?) ON CONFLICT(user_id, provider) DO
        UPDATE SET api_key=excluded.api_key
        """,
        (user_id, provider, api_key),
    )
    con.commit()
    con.close()


def get_provider_key(user_id: str, provider: str) -> str | None:
    con = _db_connect()
    cur = con.cursor()
    cur.execute(
        "SELECT api_key FROM user_provider_keys WHERE user_id = ? AND provider = ?",
        (user_id, provider),
    )
    row = cur.fetchone()
    con.close()
    return row[0] if row else None


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
            print("[LLM ERROR]", type(e).__name__, str(e))
            answer = "Error generating response"

    # save assistant message
    log_message(ctx.channel.id, ctx.author.id, "assistant", answer, ctx.guild.id if ctx.guild else None)

    # discord message limit safeguard
    if len(answer) > 1900:
        answer = answer[:1900] + "…"
    await ctx.reply(answer)


if __name__ == "__main__":
    if not DISCORD_TOKEN:  # Token error
        raise RuntimeError("Missing DISCORD_TOKEN")
    init_db()
    bot.run(DISCORD_TOKEN)
