import os
import uuid

import discord
from discord.ext import commands
from dotenv import load_dotenv


load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
BOT_MASTER_KEY = os.getenv("BOT_MASTER_KEY")
DISCORD_GUILD_ID = os.getenv("DISCORD_GUILD_ID")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
BOT_SESSION_ID = str(uuid.uuid4())


def _int_env(name: str, default: int) -> int:
    value = (os.getenv(name) or "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError as e:
        raise RuntimeError(f"{name} must be an integer") from e

intents = discord.Intents.default()
intents.message_content = True
intents.members = True

bot = commands.Bot(command_prefix="!", intents=intents, help_command=None)

DB_PATH = os.getenv("DB_PATH", "bot.db").strip() or "bot.db"
PENDING_MEMBER_CONFIRMATIONS: dict[tuple[int, int], dict] = {}
LAST_SOLVE_ARTIFACTS: dict[tuple[int, int], dict[str, str]] = {}
ARTIFACT_HISTORY: dict[tuple[int, int], list[dict[str, str]]] = {}
GUILD_ID = _int_env("DISCORD_GUILD_ID", 0)
