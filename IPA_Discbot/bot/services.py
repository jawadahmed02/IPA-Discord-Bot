import asyncio
import difflib
import json
import re
import traceback

import discord
from discord import app_commands
from discord.ext import commands

from ..mcp_client import (
    close_mcp_servers,
    connect_mcp_servers,
    list_all_mcp_tools,
    solve_pddl,
)
from .llm_helpers import (
    _all_llm_model_ids,
    _llm_classify_confirmation_reply,
    _llm_classify_member_request,
    llm_reply,
)
from .config import GUILD_ID, MODEL, PENDING_MEMBER_CONFIRMATIONS, bot
from .storage import (
    get_recent_context,
    get_user_model,
    log_message,
    save_provider_key,
    set_user_model,
)


def _truncate_discord_message(text: str, limit: int = 1900) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _split_discord_message(text: str, limit: int = 1900) -> list[str]:
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    current_length = 0

    for line in text.splitlines():
        addition = len(line) + (1 if current else 0)
        if current and current_length + addition > limit:
            chunks.append("\n".join(current))
            current = [line]
            current_length = len(line)
        else:
            current.append(line)
            current_length += addition

    if current:
        chunks.append("\n".join(current))

    return chunks


def _summarize_tool_description(description: str, limit: int = 80) -> str:
    summary = " ".join((description or "").split())
    if not summary:
        return "No description provided."
    if len(summary) <= limit:
        return summary
    return summary[: limit - 3].rstrip() + "..."


def _format_mcp_tools_message(tool_map: dict[str, list[dict[str, str]]]) -> str:
    lines: list[str] = []

    for server_name in ("paas", "l2p"):
        lines.append(f"{server_name.upper()} tools:")
        tools = tool_map.get(server_name, [])
        if not tools:
            lines.append("- none")
        for tool in tools:
            name = tool.get("name", "unknown-tool")
            description = _summarize_tool_description(tool.get("description", ""))
            lines.append(f"- `{name}`: {description}")
        lines.append("")

    return "\n".join(lines).strip()


async def _read_text_attachment(attachment: discord.Attachment) -> str:
    data = await attachment.read()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as e:
        raise RuntimeError(f"{attachment.filename} is not valid UTF-8 text.") from e


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
            attachment
            for attachment in attachments
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


def _normalize_member_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", value.lower())).strip()


def _member_name_variants(member: discord.Member) -> list[str]:
    variants = [member.display_name, member.name]
    global_name = getattr(member, "global_name", None)
    if global_name:
        variants.append(global_name)
    return [variant for variant in variants if variant]


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
        substring_bonus = (
            0.2
            if normalized_query in normalized_variant or normalized_variant in normalized_query
            else 0.0
        )
        prefix_bonus = 0.08 if normalized_variant.startswith(normalized_query) else 0.0

        best_score = max(best_score, ratio + token_bonus + substring_bonus + prefix_bonus)

    return best_score


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
            await message.reply(
                "I couldn't find another close match, so I'll stop here.",
                mention_author=False,
            )
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
            response = (
                f"{member.mention} is the closest match for `{requested_name}`, "
                "but I couldn't add them to this thread."
            )

    await message.reply(response, mention_author=False)
    return True


def _looks_like_solve_request(message: discord.Message) -> bool:
    if len(message.attachments) < 2:
        return False

    text = (message.content or "").lower()
    solve_words = ("solve", "plan", "planner", "pddl")

    if any(word in text for word in solve_words):
        return True

    filenames = " ".join(attachment.filename.lower() for attachment in message.attachments)
    return "domain" in filenames and "problem" in filenames


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

    await message.reply(
        ". ".join(parts) or "I couldn't add anyone to this thread.",
        mention_author=False,
    )
    return True


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
    should_add_to_thread = bool(decision["should_add_to_thread"]) and isinstance(
        message.channel, discord.Thread
    )
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


async def _close_bot_with_mcp_cleanup():
    await close_mcp_servers()
    await commands.Bot.close(bot)


bot.close = _close_bot_with_mcp_cleanup


@bot.tree.command(name="models", description="List available models")
async def models_cmd(interaction: discord.Interaction):
    ids = _all_llm_model_ids()
    if not ids:
        await interaction.response.send_message(
            "No models found. Install an llm provider plugin.", ephemeral=True
        )
        return

    text = "Available models:\n" + "\n".join(f"- {mid}" for mid in ids)
    await interaction.response.send_message(text[:1900], ephemeral=True)


async def model_autocomplete(
    interaction: discord.Interaction, current: str
) -> list[app_commands.Choice[str]]:
    ids = _all_llm_model_ids()
    current_lower = (current or "").lower()

    matches = [mid for mid in ids if current_lower in mid.lower()]
    matches = matches[:25]

    return [app_commands.Choice(name=mid, value=mid) for mid in matches]


@bot.tree.command(name="use", description="Choose the model this bot will use for you")
@app_commands.describe(model_id="Pick from /models")
@app_commands.autocomplete(model_id=model_autocomplete)
async def use_cmd(interaction: discord.Interaction, model_id: str):
    ids = set(_all_llm_model_ids())
    if model_id not in ids:
        await interaction.response.send_message(
            "Unknown model_id. Use /models.", ephemeral=True
        )
        return

    set_user_model(str(interaction.user.id), model_id)
    await interaction.response.send_message(f"Now using: {model_id}", ephemeral=True)


@bot.tree.command(name="setkey", description="Set your API key for a provider")
@app_commands.describe(provider="openai | gemini | anthropic", api_key="Your API key")
async def setkey_cmd(interaction: discord.Interaction, provider: str, api_key: str):
    provider = provider.strip().lower()
    if provider not in ("openai", "gemini", "anthropic"):
        await interaction.response.send_message(
            "Unknown provider. Use: openai | gemini | anthropic",
            ephemeral=True,
        )
        return

    save_provider_key(
        user_id=str(interaction.user.id),
        provider=provider,
        api_key=api_key.strip(),
    )
    await interaction.response.send_message(
        f"Saved key for {provider}.",
        ephemeral=True,
    )


@bot.event
async def on_ready():
    await connect_mcp_servers()
    print(f"Logged in as {bot.user} (id={bot.user.id})")
    guild = discord.Object(id=GUILD_ID)
    bot.tree.copy_global_to(guild=guild)
    synced = await bot.tree.sync(guild=guild)
    print(f"Synced {len(synced)} commands to guild {GUILD_ID}.")


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
            answer = await llm_reply(
                selected_model, context, user_id=str(message.author.id)
            )
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
            await ctx.reply(
                _truncate_discord_message(f"Solve failed: {type(e).__name__}: {e}")
            )
            return

    await ctx.reply(_truncate_discord_message(f"```text\n{result.strip()}\n```"))


@bot.command()
async def tools(ctx: commands.Context):
    async with ctx.typing():
        try:
            tool_map = await list_all_mcp_tools()
        except Exception as e:
            traceback.print_exc()
            await ctx.reply(
                _truncate_discord_message(
                    f"Tool listing failed: {type(e).__name__}: {e}"
                )
            )
            return

    messages = _split_discord_message(_format_mcp_tools_message(tool_map))
    await ctx.reply(messages[0])
    for chunk in messages[1:]:
        await ctx.send(chunk)
