import os
import uuid

import discord
from discord.ext import commands
from dotenv import load_dotenv


load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BOT_MASTER_KEY = os.getenv("BOT_MASTER_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
BOT_SESSION_ID = str(uuid.uuid4())

intents = discord.Intents.default()
intents.message_content = True
intents.members = True

bot = commands.Bot(command_prefix="!", intents=intents)

DB_PATH = "bot.db"
PENDING_MEMBER_CONFIRMATIONS: dict[tuple[int, int], dict] = {}
GUILD_ID = 1376609949114699886
