import os
import sqlite3
import yaml

from datetime import datetime

import discord
from discord.ext import commands
from discord import app_commands

from dotenv import load_dotenv

from openai import AsyncOpenAI

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")  # pick what your account supports

intents = discord.Intents.default() # default discord events enabled
intents.message_content = True # conversation logging

bot = commands.Bot(command_prefix="!", intents=intents)
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

DB_PATH = "bot.db" # database path

# table parameters below are as listed:
# unique id
# timestamp
# conv server
# conv channel
# conv user
# conv role
# msg content

def init_db():
    con = sqlite3.connect(DB_PATH) # opens path to db
    cur = con.cursor() # cursor is used for running SQL cmds
    cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            guild_id TEXT,
            channel_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL
        )
    """)
    con.commit()
    con.close()

def log_message(channel_id: int, user_id: int, role: str, content: str, guild_id: int | None): # conv. logging
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "INSERT INTO messages (ts, guild_id, channel_id, user_id, role, content) VALUES (?, ?, ?, ?, ?, ?)",
        (datetime.utcnow().isoformat(), str(guild_id) if guild_id else None, str(channel_id), str(user_id), role, content)
    )
    con.commit()
    con.close()

def get_recent_context(channel_id: int, user_id: int, limit: int = 12): # conv. context from channel and user
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        """
        SELECT role, content
        FROM messages
        WHERE channel_id = ? AND user_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (str(channel_id), str(user_id), limit)
    )
    rows = cur.fetchall()
    con.close()
    rows.reverse() # (oldest => newest) instead of (newest => oldest) for chronological order
    return [{"role": r, "content": c} for (r, c) in rows]

async def llm_reply(context_messages: list[dict]) -> str: # responsible for LLM response
    # Discord bot personality
    messages = [{"role": "system", "content": "You are a helpful planning assistant inside Discord. Keep answers concise."}]
    messages.extend(context_messages)

    resp = await client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )
    return resp.choices[0].message.content.strip()

# -------------------------------------- Model change features start here ----------------------------------------------
"""
@bot.tree.command(name="model", description="change current model")
async def model_cmd(interaction: discord.Interaction, model: str):
    global curr_model

    if model == curr_model:
        output = f"current model: '{curr_model}'"

    else:
        MODEL = curr_model
        output = f"current model: '{MODEL}'"
"""

# -------------------------------------- Model change features end  here -----------------------------------------------

@bot.event
async def on_ready(): # startup confirmation log
    print(f"Logged in as {bot.user} (id={bot.user.id})")

@bot.command() # Primary conversation comman
async def chat(ctx: commands.Context, *, text: str):
    # save user message
    log_message(ctx.channel.id, ctx.author.id, "user", text, ctx.guild.id if ctx.guild else None)

    async with ctx.typing():
        context = get_recent_context(ctx.channel.id, ctx.author.id, limit=12)
        answer = await llm_reply(context)

    # save assistant message
    log_message(ctx.channel.id, ctx.author.id, "assistant", answer, ctx.guild.id if ctx.guild else None)

    # discord message limit safeguard
    if len(answer) > 1900:
        answer = answer[:1900] + "…"
    await ctx.reply(answer)

if __name__ == "__main__":
    if not DISCORD_TOKEN or not OPENAI_API_KEY: # Token error
        raise RuntimeError("Missing DISCORD_TOKEN or OPENAI_API_KEY in environment/.env")
    init_db()
    bot.run(DISCORD_TOKEN)
