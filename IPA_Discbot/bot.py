import os
import sqlite3
import llm
import asyncio
import threading
import json
import difflib
import re
import uuid
import traceback

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
BOT_SESSION_ID = str(uuid.uuid4())

intents = discord.Intents.default()  # default discord events enabled
intents.message_content = True  # conversation logging
intents.members = True  # fuzzy member lookup for conversational mentions

bot = commands.Bot(command_prefix="!", intents=intents)

DB_PATH = "bot.db"  # database path
PENDING_MEMBER_CONFIRMATIONS: dict[tuple[int, int], dict] = {}


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


# Adds a session id column to old message tables so each bot startup can be tracked.
def migrate_messages_session_column_if_needed():
    con = _db_connect()
    try:
        cur = con.cursor()
        cur.execute("PRAGMA table_info(messages)")
        columns = cur.fetchall()
        column_names = {col[1] for col in columns}

        if "session_id" not in column_names:
            print("[DB] Adding session_id column to messages...")
            cur.execute("ALTER TABLE messages ADD COLUMN session_id TEXT")

        cur.execute(
            """
            UPDATE messages
            SET session_id = 'legacy'
            WHERE session_id IS NULL OR TRIM(session_id) = ''
            """
        )
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
            session_id TEXT NOT NULL,
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
    migrate_messages_session_column_if_needed()
    print(f"[DB] Bot session started: {BOT_SESSION_ID}")



def log_message(channel_id: int, user_id: int, role: str, content: str, guild_id: int | None):  # conv. logging for specific user
    con = _db_connect()
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO messages (ts, guild_id, channel_id, user_id, session_id, role, content)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now(UTC).isoformat(),
            str(guild_id) if guild_id else None,
            str(channel_id),
            str(user_id),
            BOT_SESSION_ID,
            role,
            content,
        )
    )
    con.commit()
    con.close()


def get_recent_context(
    user_id: int,
    guild_id: int | None,
    channel_id: int | None = None,
    limit: int = 1000,
):  # conv. context from this user across sessions, preferring the active channel
    con = _db_connect()
    cur = con.cursor()
    cur.execute(
        """
        SELECT role, content, session_id, channel_id
        FROM messages
        WHERE user_id = ?
          AND (
                (? IS NULL AND guild_id IS NULL)
                OR guild_id = ?
              )
        ORDER BY
            CASE WHEN channel_id = ? THEN 0 ELSE 1 END,
            id DESC
        LIMIT ?
        """,
        (
            str(user_id),
            str(guild_id) if guild_id is not None else None,
            str(guild_id) if guild_id is not None else None,
            str(channel_id) if channel_id is not None else None,
            limit,
        ),
    )
    rows = cur.fetchall()
    con.close()
    rows.reverse()  # (oldest => newest) instead of (newest => oldest) for chronological order

    context: list[dict] = []
    previous_session_id: str | None = None
    previous_channel_id: str | None = None

    for role, content, session_id, row_channel_id in rows:
        if session_id != previous_session_id:
            context.append(
                {
                    "role": "system",
                    "content": f"Conversation session: {session_id}",
                }
            )
            previous_session_id = session_id

        if row_channel_id != previous_channel_id:
            context.append(
                {
                    "role": "system",
                    "content": f"Discord channel: {row_channel_id}",
                }
            )
            previous_channel_id = row_channel_id

        context.append({"role": role, "content": content})

    return context

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


# Returns the shared system prompt that tells the model to use persisted Discord history.
def _conversation_system_prompt() -> str:
    return (
        "You are a helpful planning assistant inside Discord. "
        "Keep answers concise. "
        "You are given persisted conversation history for this user, and that history can include previous bot sessions. "
        "Use facts, variables, preferences, and prior commitments from the provided history when answering. "
        "If the needed information appears in the provided history, do not say you cannot access previous sessions or that memory was lost after a restart. "
        "Only say information is unavailable when it truly does not appear in the supplied conversation history."
    )


# Extracts the first JSON object from a model response so tool-routing stays robust.
def _parse_llm_json_object(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in LLM response.")
    return json.loads(text[start:end + 1])


def _llm_reply_sync(model_id: str, context_messages: list[dict]) -> str:
    system_prompt = _conversation_system_prompt()

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


# Runs a one-off LLM prompt for a user while safely swapping provider credentials.
def _run_llm_prompt_for_user_sync(user_id: str, model_id: str, prompt: str, system_prompt: str) -> str:
    provider = _provider_from_model_id(model_id)
    env_key = PROVIDER_ENV.get(provider or "")

    user_key = None
    if env_key:
        user_key = get_provider_key(user_id, provider)
        if not user_key:
            raise RuntimeError(f"No API key set for {provider}. Use /setkey {provider} <key>.")

    with LLM_ENV_LOCK:
        old = os.environ.get(env_key) if env_key else None
        try:
            if env_key and user_key:
                os.environ[env_key] = user_key

            model = llm.get_model(model_id)
            response = model.prompt(prompt, system=system_prompt)
            return response.text().strip()
        finally:
            if env_key:
                if old is None:
                    os.environ.pop(env_key, None)
                else:
                    os.environ[env_key] = old


def _run_llm_for_user_sync(user_id: str, model_id: str, context_messages: list[dict]) -> str:
    provider = _provider_from_model_id(model_id)
    env_key = PROVIDER_ENV.get(provider or "")

    transcript = _build_transcript(context_messages)
    system_prompt = _conversation_system_prompt()

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


# Asks the selected model to classify a short control message as confirm/reject/cancel/other.
async def _llm_classify_confirmation_reply(message: discord.Message) -> str:
    selected_model = get_user_model(str(message.author.id)) or MODEL
    prompt = (
        "Classify the user's reply to a confirmation question.\n"
        "Return only JSON with this schema:\n"
        "{\"reply_type\":\"confirm|reject|cancel|other\"}\n"
        f"User reply: {json.dumps(message.content or '')}"
    )
    system_prompt = (
        "You classify short confirmation replies for a Discord bot. "
        "Infer intent from meaning, not fixed keywords. "
        "Return only valid JSON."
    )

    raw = await asyncio.to_thread(
        _run_llm_prompt_for_user_sync,
        str(message.author.id),
        selected_model,
        prompt,
        system_prompt,
    )
    data = _parse_llm_json_object(raw)
    reply_type = str(data.get("reply_type", "other")).strip().lower()
    if reply_type not in {"confirm", "reject", "cancel", "other"}:
        return "other"
    return reply_type


# Asks the model whether a natural-language message is requesting a member mention/thread add.
async def _llm_classify_member_request(message: discord.Message) -> dict:
    selected_model = get_user_model(str(message.author.id)) or MODEL
    mentioned_users = [
        {
            "id": user.id,
            "name": getattr(user, "display_name", None) or user.name,
        }
        for user in message.mentions
    ]
    prompt = (
        "Decide whether this Discord message is asking the bot to mention a server member or add mentioned users to a thread.\n"
        "Return only JSON with this schema:\n"
        "{\"intent\":\"member_lookup|thread_add_mentions|none\",\"requested_name\":\"string\",\"should_add_to_thread\":true}\n"
        f"Is thread: {json.dumps(isinstance(message.channel, discord.Thread))}\n"
        f"Mentioned users: {json.dumps(mentioned_users)}\n"
        f"Message: {json.dumps(message.content or '')}"
    )
    system_prompt = (
        "You classify Discord bot control requests. "
        "Infer intent from the user's wording, even when synonyms or unusual phrasing are used. "
        "Use thread_add_mentions only when the message is asking to add already-mentioned users to the current thread. "
        "Use member_lookup when the user refers to someone by name and the bot should resolve and mention them. "
        "Set should_add_to_thread true when the user wants the matched person brought into the current thread. "
        "Return only valid JSON."
    )

    raw = await asyncio.to_thread(
        _run_llm_prompt_for_user_sync,
        str(message.author.id),
        selected_model,
        prompt,
        system_prompt,
    )
    data = _parse_llm_json_object(raw)
    intent = str(data.get("intent", "none")).strip().lower()
    if intent not in {"member_lookup", "thread_add_mentions", "none"}:
        intent = "none"

    requested_name = str(data.get("requested_name", "") or "").strip()
    should_add_to_thread = bool(data.get("should_add_to_thread", False))
    return {
        "intent": intent,
        "requested_name": requested_name,
        "should_add_to_thread": should_add_to_thread,
    }


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


# Normalizes member names and user text so fuzzy matching is more reliable.
def _normalize_member_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", value.lower())).strip()


# Collects the name variants we can use when matching a Discord member.
def _member_name_variants(member: discord.Member) -> list[str]:
    variants = [member.display_name, member.name]
    global_name = getattr(member, "global_name", None)
    if global_name:
        variants.append(global_name)
    return [variant for variant in variants if variant]


# Scores how closely a guild member matches the requested natural-language name.
def _score_member_match(query: str, member: discord.Member) -> float:
    normalized_query = _normalize_member_text(query)
    if not normalized_query:
        return 0.0

    best_score = 0.0
    query_tokens = set(normalized_query.split())

    for variant in _member_name_variants(member):
        normalized_variant = _normalize_member_text(variant)
        if not normalized_variant:
            continue

        ratio = difflib.SequenceMatcher(None, normalized_query, normalized_variant).ratio()
        token_overlap = len(query_tokens & set(normalized_variant.split()))
        token_bonus = 0.12 * token_overlap
        substring_bonus = 0.2 if normalized_query in normalized_variant or normalized_variant in normalized_query else 0.0
        prefix_bonus = 0.08 if normalized_variant.startswith(normalized_query) else 0.0

        best_score = max(best_score, ratio + token_bonus + substring_bonus + prefix_bonus)

    return best_score


# Returns likely member matches ordered from best to worst for confirmation cycling.
async def _rank_matching_members(guild: discord.Guild, query: str) -> list[discord.Member]:
    exact_member = guild.get_member_named(query)
    if exact_member is not None:
        return [exact_member]

    members = list(guild.members)
    if not members:
        try:
            members = [member async for member in guild.fetch_members(limit=None)]
        except discord.HTTPException:
            return []

    scored_members: list[tuple[float, discord.Member]] = []
    for member in members:
        score = _score_member_match(query, member)
        if score >= 0.45:
            scored_members.append((score, member))

    scored_members.sort(
        key=lambda item: (
            -item[0],
            len(_normalize_member_text(item[1].display_name or item[1].name)),
        )
    )
    return [member for _, member in scored_members]


# Handles follow-up yes/no/cancel replies for pending fuzzy member confirmations.
async def _handle_member_confirmation_response(message: discord.Message) -> bool:
    key = (message.channel.id, message.author.id)
    pending = PENDING_MEMBER_CONFIRMATIONS.get(key)
    if pending is None:
        return False

    try:
        reply_type = await _llm_classify_confirmation_reply(message)
    except Exception as e:
        print("[LLM CONFIRMATION CLASSIFIER ERROR]", type(e).__name__, e)
        return False

    if reply_type == "cancel":
        PENDING_MEMBER_CONFIRMATIONS.pop(key, None)
        await message.reply("Okay, I stopped the member lookup.", mention_author=False)
        return True

    if reply_type == "reject":
        candidate_ids = pending["candidate_ids"]
        current_index = pending["current_index"] + 1
        guild = message.guild

        if guild is None:
            PENDING_MEMBER_CONFIRMATIONS.pop(key, None)
            await message.reply("Okay, I won't mention them.", mention_author=False)
            return True

        next_member: discord.Member | None = None
        while current_index < len(candidate_ids):
            next_member = guild.get_member(candidate_ids[current_index])
            if next_member is not None:
                break
            current_index += 1

        if next_member is None:
            PENDING_MEMBER_CONFIRMATIONS.pop(key, None)
            await message.reply("I couldn't find another close match, so I'll stop here.", mention_author=False)
            return True

        pending["current_index"] = current_index
        action = "mention them"
        if pending["should_add_to_thread"]:
            action = "mention them and add them to this thread"

        await message.reply(
            f"Okay, how about {next_member.mention}? Reply `yes` to {action}, or `no` if that's not the right user either.",
            mention_author=False,
        )
        return True

    if reply_type != "confirm":
        return False

    PENDING_MEMBER_CONFIRMATIONS.pop(key, None)

    guild = message.guild
    if guild is None:
        return True

    member = guild.get_member(pending["candidate_ids"][pending["current_index"]])
    if member is None:
        await message.reply("I couldn't find that member anymore.", mention_author=False)
        return True

    requested_name = pending["requested_name"]
    should_add_to_thread = pending["should_add_to_thread"]
    response = f"{member.mention}"

    if should_add_to_thread and isinstance(message.channel, discord.Thread):
        try:
            await message.channel.add_user(member)
            response = f"{member.mention} added to this thread."
        except discord.HTTPException:
            response = f"{member.mention} is the closest match for `{requested_name}`, but I couldn't add them to this thread."

    await message.reply(response, mention_author=False)
    return True


# Checks whether a natural-language message should be routed to the PDDL solver.
def _looks_like_solve_request(message: discord.Message) -> bool:
    if len(message.attachments) < 2:
        return False

    text = (message.content or "").lower()
    solve_words = ("solve", "plan", "planner", "pddl")

    if any(word in text for word in solve_words):
        return True

    filenames = " ".join(attachment.filename.lower() for attachment in message.attachments)
    return "domain" in filenames and "problem" in filenames


# Adds explicitly mentioned users to the active thread when the request matches.
async def _handle_thread_add_request(message: discord.Message) -> bool:
    if not isinstance(message.channel, discord.Thread):
        return False

    if not message.mentions:
        return False

    try:
        decision = await _llm_classify_member_request(message)
    except Exception as e:
        print("[LLM MEMBER REQUEST CLASSIFIER ERROR]", type(e).__name__, e)
        return False

    if decision["intent"] != "thread_add_mentions":
        return False

    thread = message.channel
    added_users: list[str] = []
    failed_users: list[str] = []

    async with thread.typing():
        for user in message.mentions:
            if user.bot:
                failed_users.append(user.mention)
                continue

            try:
                await thread.add_user(user)
                added_users.append(user.mention)
            except discord.HTTPException:
                failed_users.append(user.mention)

    parts: list[str] = []
    if added_users:
        parts.append("Added to thread: " + ", ".join(added_users))
    if failed_users:
        parts.append("Couldn't add: " + ", ".join(failed_users))

    await message.reply(". ".join(parts) or "I couldn't add anyone to this thread.", mention_author=False)
    return True


# Resolves conversational name requests into a confirmed Discord member mention.
async def _handle_member_lookup_request(message: discord.Message) -> bool:
    if message.guild is None:
        return False

    if message.mentions:
        return False

    try:
        decision = await _llm_classify_member_request(message)
    except Exception as e:
        print("[LLM MEMBER REQUEST CLASSIFIER ERROR]", type(e).__name__, e)
        return False

    if decision["intent"] != "member_lookup":
        return False

    requested_name = decision["requested_name"]
    if not requested_name:
        return False

    candidates = await _rank_matching_members(message.guild, requested_name)
    if not candidates:
        await message.reply(
            f"I couldn't find a close match for `{requested_name}` in this server.",
            mention_author=False,
        )
        return True

    member = candidates[0]
    should_add_to_thread = bool(decision["should_add_to_thread"]) and isinstance(message.channel, discord.Thread)
    PENDING_MEMBER_CONFIRMATIONS[(message.channel.id, message.author.id)] = {
        "candidate_ids": [candidate.id for candidate in candidates],
        "current_index": 0,
        "requested_name": requested_name,
        "should_add_to_thread": should_add_to_thread,
    }

    action = "mention them"
    if should_add_to_thread:
        action = "mention them and add them to this thread"

    await message.reply(
        f"Did you mean {member.mention}? Reply `yes` to {action}, or `no` to cancel.",
        mention_author=False,
    )
    return True


# Runs the planner from a normal message when the user provides PDDL attachments.
async def _handle_solve_request(message: discord.Message) -> bool:
    if not _looks_like_solve_request(message):
        return False

    async with message.channel.typing():
        try:
            domain_text, problem_text = await _extract_pddl_attachments(message)
            result = await solve_pddl(domain_text, problem_text)
        except Exception as e:
            traceback.print_exc()
            await message.reply(
                _truncate_discord_message(f"Solve failed: {type(e).__name__}: {e}"),
                mention_author=False,
            )
            return True

    await message.reply(
        _truncate_discord_message(f"```text\n{result.strip()}\n```"),
        mention_author=False,
    )
    return True


# -------------------------------------- Model change features end  here -----------------------------------------------

GUILD_ID = 1376609949114699886 # uBots server id

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user} (id={bot.user.id})")
    guild = discord.Object(id=GUILD_ID)
    bot.tree.copy_global_to(guild=guild)
    synced = await bot.tree.sync(guild=guild)
    print(f"Synced {len(synced)} commands to guild {GUILD_ID}.")


# Routes normal Discord messages through confirmations, tools, and then LLM chat.
async def _handle_conversation_message(message: discord.Message):
    if await _handle_member_confirmation_response(message):
        return

    if await _handle_thread_add_request(message):
        return

    if await _handle_member_lookup_request(message):
        return

    if await _handle_solve_request(message):
        return

    text = (message.content or "").strip()
    if not text:
        return

    log_message(
        message.channel.id,
        message.author.id,
        "user",
        text,
        message.guild.id if message.guild else None,
    )

    async with message.channel.typing():
        context = get_recent_context(
            user_id=message.author.id,
            guild_id=message.guild.id if message.guild else None,
            channel_id=message.channel.id,
        )
        selected_model = get_user_model(str(message.author.id)) or MODEL
        try:
            answer = await llm_reply(selected_model, context, user_id=str(message.author.id))
        except Exception as e:
            print("[LLM ERROR]", type(e).__name__)
            answer = "Error generating response"

    log_message(
        message.channel.id,
        message.author.id,
        "assistant",
        answer,
        message.guild.id if message.guild else None,
    )

    await message.reply(_truncate_discord_message(answer), mention_author=False)


@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    ctx = await bot.get_context(message)
    if ctx.valid:
        await bot.process_commands(message)
        return

    await _handle_conversation_message(message)


@bot.command()
async def chat(ctx: commands.Context, *, topic: str | None = None):
    if ctx.guild is None:
        await ctx.reply("Thread creation only works in a server channel.")
        return

    if isinstance(ctx.channel, discord.Thread):
        await ctx.reply("This is already a thread.")
        return

    if not isinstance(ctx.channel, discord.TextChannel):
        await ctx.reply("Create a chat thread from a regular server text channel.")
        return

    thread_name = (topic or f"Chat with {ctx.author.display_name}").strip()
    thread_name = thread_name[:100] or f"Chat with {ctx.author.display_name}"

    try:
        thread = await ctx.channel.create_thread(
            name=thread_name,
            type=discord.ChannelType.private_thread,
            auto_archive_duration=1440,
            invitable=False,
        )
    except discord.HTTPException:
        await ctx.reply("I couldn't create a thread here. Check my thread permissions.")
        return

    try:
        await thread.add_user(ctx.author)
    except discord.HTTPException:
        pass

    await ctx.reply(f"Started a chat thread: {thread.mention}")

@bot.command()
async def solve(ctx: commands.Context):
    async with ctx.typing():
        try:
            domain_text, problem_text = await _extract_pddl_attachments(ctx.message)
            result = await solve_pddl(domain_text, problem_text)
        except Exception as e:
            traceback.print_exc()
            await ctx.reply(_truncate_discord_message(f"Solve failed: {type(e).__name__}: {e}"))
            return

    await ctx.reply(_truncate_discord_message(f"```text\n{result.strip()}\n```"))


if __name__ == "__main__":
    if not DISCORD_TOKEN:
        raise RuntimeError("Missing DISCORD_TOKEN")
    if not BOT_MASTER_KEY:
        raise RuntimeError("Missing BOT_MASTER_KEY")

    init_db()
    bot.run(DISCORD_TOKEN)
