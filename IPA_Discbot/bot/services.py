import io
import re
import traceback

import discord
from discord import app_commands
from discord.ext import commands

from IPA_Discbot.mcp_client import (
    close_mcp_servers,
    get_mcp_tool_catalog,
    list_all_mcp_tools,
)
from .llm_helpers import _all_llm_model_ids
from .config import GUILD_ID, bot
from .parsing import (
    _normalize_artifact_type,
    _split_discord_message,
    _truncate_discord_message,
)
from .state import (
    _copy_personal_artifacts_to_shared,
    _load_runtime_artifact_state,
    _persist_artifacts_if_session_saved,
    _update_working_artifacts,
)
from .storage import (
    get_share_mode,
    get_user_model,
    is_collab_enabled,
    is_chat_enabled,
    load_saved_working_artifacts,
    save_current_session,
    save_provider_key,
    set_collab_enabled,
    set_chat_enabled,
    set_share_mode,
    set_user_model,
    user_has_any_provider_key,
)
from .workflows import (
    _format_domain_reply,
    _format_help_message,
    _format_mcp_tools_message,
    _format_problem_reply,
    _format_single_server_tools_message,
    handle_conversation_message,
    run_autovalidate_request,
    run_domain_request,
    run_edit_domain_request,
    run_edit_plan_request,
    run_edit_problem_request,
    run_explain_artifact_request,
    run_files_request,
    run_plan_request,
    run_problem_request,
    run_show_artifact_request,
    run_undo_request,
    run_validate_domain_request,
    run_validate_plan_request,
    run_validate_request,
    run_validate_task_request,
)


def _safe_pddl_name(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", (value or "").strip()).strip("_")
    return cleaned or fallback


def _text_file(text: str, filename: str) -> discord.File:
    return discord.File(io.BytesIO(text.encode("utf-8")), filename=filename)


async def _close_bot_with_mcp_cleanup():
    await close_mcp_servers()
    await commands.Bot.close(bot)


bot.close = _close_bot_with_mcp_cleanup


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------

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
    matches = [mid for mid in ids if current_lower in mid.lower()][:25]
    return [app_commands.Choice(name=mid, value=mid) for mid in matches]


@bot.tree.command(name="use", description="Choose the model this bot will use for you")
@app_commands.describe(model_id="Pick from /models")
@app_commands.autocomplete(model_id=model_autocomplete)
async def use_cmd(interaction: discord.Interaction, model_id: str):
    if model_id not in set(_all_llm_model_ids()):
        await interaction.response.send_message("Unknown model_id. Use /models.", ephemeral=True)
        return
    set_user_model(str(interaction.user.id), model_id)
    await interaction.response.send_message(f"Now using: {model_id}", ephemeral=True)


@bot.tree.command(name="setkey", description="Set your API key for a provider")
@app_commands.describe(provider="openai | gemini | anthropic", api_key="Your API key")
async def setkey_cmd(interaction: discord.Interaction, provider: str, api_key: str):
    provider = provider.strip().lower()
    if provider not in ("openai", "gemini", "anthropic"):
        await interaction.response.send_message(
            "Unknown provider. Use: openai | gemini | anthropic", ephemeral=True
        )
        return
    save_provider_key(user_id=str(interaction.user.id), provider=provider, api_key=api_key.strip())
    await interaction.response.send_message(f"Saved key for {provider}.", ephemeral=True)


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

@bot.event
async def on_ready():
    saved_artifacts, saved_history = load_saved_working_artifacts()
    _load_runtime_artifact_state(saved_artifacts, saved_history)
    print(f"Logged in as {bot.user} (id={bot.user.id})")
    print("[MCP] Startup uses lazy MCP connections. Planning commands will connect on first use.")
    guild = discord.Object(id=GUILD_ID)
    bot.tree.copy_global_to(guild=guild)
    synced = await bot.tree.sync(guild=guild)
    print(f"Synced {len(synced)} commands to guild {GUILD_ID}.")


@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    ctx = await bot.get_context(message)
    if ctx.valid:
        await bot.process_commands(message)
        return

    channel_id = str(message.channel.id)
    if not is_collab_enabled(channel_id) and not is_chat_enabled(
        str(message.author.id), channel_id
    ):
        return

    await handle_conversation_message(message)


# ---------------------------------------------------------------------------
# Prefix commands
# ---------------------------------------------------------------------------

@bot.command()
async def thread(ctx: commands.Context, *, topic: str | None = None):
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
        new_thread = await ctx.channel.create_thread(
            name=thread_name,
            type=discord.ChannelType.private_thread,
            auto_archive_duration=1440,
            invitable=False,
        )
    except discord.HTTPException:
        await ctx.reply("I couldn't create a thread here. Check my thread permissions.")
        return

    try:
        await new_thread.add_user(ctx.author)
    except discord.HTTPException:
        pass

    await ctx.reply(f"Started a chat thread: {new_thread.mention}")


@bot.command()
async def help(ctx: commands.Context):
    messages = _split_discord_message(_format_help_message())
    await ctx.reply(messages[0])
    for chunk in messages[1:]:
        await ctx.send(chunk)


@bot.command()
async def chat(ctx: commands.Context):
    user_id = str(ctx.author.id)
    channel_id = str(ctx.channel.id)
    enabled = not is_chat_enabled(user_id, channel_id)
    set_chat_enabled(user_id, channel_id, enabled)
    if enabled:
        await ctx.reply("LLM chat is now `on` in this channel for you.")
    else:
        await ctx.reply(
            "LLM chat is now `off` in this channel for you. Your normal messages here will no longer call the bot until you run `!chat` again."
        )


@bot.command()
async def collab(ctx: commands.Context):
    channel_id = str(ctx.channel.id)
    enabled = not is_collab_enabled(channel_id)
    set_collab_enabled(channel_id, enabled)
    if enabled:
        _copy_personal_artifacts_to_shared(ctx.channel.id, ctx.author.id)
        await ctx.reply(
            "Collaboration mode is now `on` in this channel. Everyone here can use shared chat context and work on the same domain, problem, and plan."
        )
    else:
        await ctx.reply(
            "Collaboration mode is now `off` in this channel. Normal chat and planning artifacts are separate for each user again."
        )


@bot.command(name="plan")
async def plan_cmd(ctx: commands.Context, *, request: str | None = None):
    async with ctx.typing():
        try:
            reply_text, files = await run_plan_request(ctx.message, request)
        except Exception as e:
            traceback.print_exc()
            await ctx.reply(_truncate_discord_message(f"Solve failed: {type(e).__name__}: {e}"))
            return
    await ctx.reply(reply_text, files=files)


@bot.command(name="domain")
async def domain_cmd(ctx: commands.Context, *, request: str | None = None):
    request_text = (request or "").strip()
    if not request_text:
        await ctx.reply("Use `!domain <natural language request>`.")
        return

    async with ctx.typing():
        try:
            domain_name, domain_text = await run_domain_request(ctx.message, request_text)
            _update_working_artifacts(ctx.message, domain=domain_text, domain_name=domain_name)
        except Exception as e:
            traceback.print_exc()
            await ctx.reply(
                _truncate_discord_message(f"Domain generation failed: {type(e).__name__}: {e}")
            )
            return
    reply_text = _format_domain_reply(domain_name, domain_text)
    messages = _split_discord_message(reply_text)
    await ctx.reply(messages[0])
    for chunk in messages[1:]:
        await ctx.send(chunk)


@bot.command(name="problem")
async def problem_cmd(ctx: commands.Context, *, request: str | None = None):
    request_text = (request or "").strip()
    if not request_text:
        await ctx.reply("Use `!problem <natural language request>`.")
        return

    async with ctx.typing():
        try:
            problem_name, problem_text = await run_problem_request(ctx.message, request_text)
            _update_working_artifacts(ctx.message, problem=problem_text, problem_name=problem_name)
        except Exception as e:
            traceback.print_exc()
            await ctx.reply(
                _truncate_discord_message(f"Problem generation failed: {type(e).__name__}: {e}")
            )
            return
    reply_text = _format_problem_reply(problem_name, problem_text)
    messages = _split_discord_message(reply_text)
    await ctx.reply(messages[0])
    for chunk in messages[1:]:
        await ctx.send(chunk)


@bot.command(name="show")
async def show_cmd(ctx: commands.Context, artifact_type: str):
    normalized = _normalize_artifact_type(artifact_type)
    if normalized is None:
        await ctx.reply("Use `!show domain`, `!show problem`, or `!show plan`.")
        return

    async with ctx.typing():
        try:
            reply_text, files = await run_show_artifact_request(ctx.message, normalized)
        except Exception as e:
            traceback.print_exc()
            await ctx.reply(_truncate_discord_message(f"Show failed: {type(e).__name__}: {e}"))
            return

    messages = _split_discord_message(reply_text)
    await ctx.reply(messages[0], files=files or None)
    for chunk in messages[1:]:
        await ctx.send(chunk)


@bot.command(name="files")
async def files_cmd(ctx: commands.Context):
    async with ctx.typing():
        try:
            reply_text, files = await run_files_request(ctx.message)
        except Exception as e:
            traceback.print_exc()
            await ctx.reply(_truncate_discord_message(f"Files failed: {type(e).__name__}: {e}"))
            return
    await ctx.reply(reply_text, files=files)


@bot.command(name="explain")
async def explain_cmd(ctx: commands.Context, artifact_type: str):
    normalized = _normalize_artifact_type(artifact_type)
    if normalized is None:
        await ctx.reply("Use `!explain domain`, `!explain problem`, or `!explain plan`.")
        return

    async with ctx.typing():
        try:
            reply_text = await run_explain_artifact_request(ctx.message, normalized)
        except Exception as e:
            traceback.print_exc()
            await ctx.reply(_truncate_discord_message(f"Explain failed: {type(e).__name__}: {e}"))
            return

    messages = _split_discord_message(reply_text)
    await ctx.reply(messages[0])
    for chunk in messages[1:]:
        await ctx.send(chunk)


@bot.command(name="edit")
async def edit_cmd(ctx: commands.Context, artifact_type: str, *, instruction: str | None = None):
    normalized = _normalize_artifact_type(artifact_type)
    if normalized is None:
        await ctx.reply("Use `!edit domain ...`, `!edit problem ...`, or `!edit plan ...`.")
        return

    edit_instruction = (instruction or "").strip()
    if not edit_instruction:
        await ctx.reply(f"Tell me how to revise the {normalized}.")
        return

    async with ctx.typing():
        try:
            if normalized == "domain":
                reply_text, files = await run_edit_domain_request(ctx.message, edit_instruction)
            elif normalized == "problem":
                reply_text, files = await run_edit_problem_request(ctx.message, edit_instruction)
            else:
                reply_text, files = await run_edit_plan_request(ctx.message, edit_instruction)
        except Exception as e:
            traceback.print_exc()
            await ctx.reply(_truncate_discord_message(f"Edit failed: {type(e).__name__}: {e}"))
            return

    messages = _split_discord_message(reply_text)
    await ctx.reply(messages[0], files=files or None)
    for chunk in messages[1:]:
        await ctx.send(chunk)


@bot.command(name="undo")
async def undo_cmd(ctx: commands.Context, artifact_type: str):
    normalized = _normalize_artifact_type(artifact_type)
    if normalized not in {"domain", "problem"}:
        await ctx.reply("Use `!undo domain` or `!undo problem`.")
        return

    async with ctx.typing():
        try:
            reply_text, files = await run_undo_request(ctx.message, normalized)
        except Exception as e:
            traceback.print_exc()
            await ctx.reply(_truncate_discord_message(f"Undo failed: {type(e).__name__}: {e}"))
            return

    messages = _split_discord_message(reply_text)
    await ctx.reply(messages[0], files=files or None)
    for chunk in messages[1:]:
        await ctx.send(chunk)


@bot.command(name="validate")
async def validate_cmd(ctx: commands.Context):
    async with ctx.typing():
        try:
            result = await run_validate_request(ctx.message)
        except Exception as e:
            traceback.print_exc()
            await ctx.reply(
                _truncate_discord_message(f"VAL validation failed: {type(e).__name__}: {e}")
            )
            return
    await ctx.reply(result)


@bot.command(name="validate_plan")
async def validate_plan_cmd(ctx: commands.Context):
    async with ctx.typing():
        try:
            result = await run_validate_plan_request(ctx.message)
        except Exception as e:
            traceback.print_exc()
            await ctx.reply(
                _truncate_discord_message(f"Plan validation failed: {type(e).__name__}: {e}")
            )
            return
    await ctx.reply(result)


@bot.command(name="validate_domain")
async def validate_domain_cmd(ctx: commands.Context):
    async with ctx.typing():
        try:
            result = await run_validate_domain_request(ctx.message)
        except Exception as e:
            traceback.print_exc()
            await ctx.reply(
                _truncate_discord_message(f"Domain validation failed: {type(e).__name__}: {e}")
            )
            return
    await ctx.reply(result)


@bot.command(name="validate_task")
async def validate_task_cmd(ctx: commands.Context):
    async with ctx.typing():
        try:
            result = await run_validate_task_request(ctx.message)
        except Exception as e:
            traceback.print_exc()
            await ctx.reply(
                _truncate_discord_message(f"Task validation failed: {type(e).__name__}: {e}")
            )
            return
    await ctx.reply(result)


@bot.command(name="autovalidate")
async def autovalidate_cmd(ctx: commands.Context):
    async with ctx.typing():
        try:
            reply_text, files = await run_autovalidate_request(ctx.message)
        except Exception as e:
            traceback.print_exc()
            await ctx.reply(
                _truncate_discord_message(f"Auto-validate failed: {type(e).__name__}: {e}")
            )
            return
    await ctx.reply(reply_text, files=files)


@bot.command(name="paastools")
async def paastools_cmd(ctx: commands.Context):
    async with ctx.typing():
        try:
            tool_catalog = await get_mcp_tool_catalog()
            paas_tools = tool_catalog.get("paas", [])
        except Exception as e:
            traceback.print_exc()
            await ctx.reply(
                _truncate_discord_message(f"PaaS tool listing failed: {type(e).__name__}: {e}")
            )
            return

    messages = _split_discord_message(_format_single_server_tools_message("paas", paas_tools))
    await ctx.reply(messages[0])
    for chunk in messages[1:]:
        await ctx.send(chunk)


@bot.command()
async def tools(ctx: commands.Context):
    async with ctx.typing():
        try:
            tool_map = await list_all_mcp_tools()
        except Exception as e:
            traceback.print_exc()
            await ctx.reply(
                _truncate_discord_message(f"Tool listing failed: {type(e).__name__}: {e}")
            )
            return

    messages = _split_discord_message(_format_mcp_tools_message(tool_map))
    await ctx.reply(messages[0])
    for chunk in messages[1:]:
        await ctx.send(chunk)


@bot.command()
async def share(ctx: commands.Context):
    user_id = str(ctx.author.id)
    if not user_has_any_provider_key(user_id):
        await ctx.reply("Set an API key first with `/setkey`, then you can toggle sharing.")
        return

    current_mode = get_share_mode(user_id)
    new_mode = "group" if current_mode == "individual" else "individual"
    set_share_mode(user_id, new_mode)

    if new_mode == "group":
        await ctx.reply(
            "Share mode is now `group`. Other users can use the bot through your saved API key when they do not have one."
        )
    else:
        await ctx.reply("Share mode is now `individual`. Only you can use your saved API key.")


@bot.command()
async def save(ctx: commands.Context):
    is_new_save = save_current_session()
    _persist_artifacts_if_session_saved()
    if is_new_save:
        await ctx.reply("Saved the current bot session. Its conversation log will be kept after restart.")
    else:
        await ctx.reply("This bot session was already saved.")
