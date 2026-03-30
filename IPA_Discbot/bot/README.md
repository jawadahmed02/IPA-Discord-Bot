# Bot Package

This package contains the Discord bot itself. It is responsible for loading runtime configuration, creating the `discord.py` bot instance, handling commands and events, storing conversation history, and routing user requests into either normal LLM chat or the MCP planning layer.

## Files

- `config.py`: shared runtime configuration, environment loading, Discord intents, and the global bot instance
- `storage.py`: SQLite access, encrypted provider key storage, message logging, and per-user model persistence
- `llm_helpers.py`: LLM prompting helpers plus lightweight classification helpers used for conversational routing
- `services.py`: Discord commands, slash commands, event handlers, message parsing, and message-to-action routing
- `__init__.py`: package entrypoint exposing `bot` and `run()`
- `__main__.py`: allows the package to be executed as a module

## Runtime Flow

1. `run()` in `__init__.py` checks required environment variables.
2. The database is initialized through `storage.py`.
3. The Discord bot created in `config.py` starts running.
4. Importing `services.py` registers the bot commands and event handlers.
5. On startup, the bot connects to both MCP backends.
6. Messages are routed into one of four main paths:
   - direct command handling
   - member or thread helper actions
   - planning requests from PDDL attachments
   - general LLM conversation with persisted context

## Main Behaviors

- Logs user and assistant messages into SQLite for cross-session context
- Lets users choose models with `/models` and `/use`
- Stores per-user provider keys with `/setkey`
- Accepts solve requests through `!solve` with either PDDL attachments or a natural-language request
- Supports `!tools` to list available MCP tools from both backends
- Supports private chat-thread creation with `!thread`
- Supports per-channel chat toggling with `!chat`
- Supports provider-key sharing with `!share` and session persistence with `!save`

## Running The Bot

From the repo root, run:

```bash
python3 -m IPA_Discbot.bot
```

Before starting the bot, make sure your environment is set up with:

- `DISCORD_TOKEN`
- `BOT_MASTER_KEY`
- `PAAS_MCP_URL`
- `L2P_MCP_URL`

If you want to use a provider-backed model for conversation or natural-language solve, also set the corresponding provider key such as `OPENAI_API_KEY`, `LLM_GEMINI_KEY`, or `ANTHROPIC_API_KEY`.

## Dependencies On Other Packages

This package depends on `IPA_Discbot.mcp_client` for planning operations. The bot layer does not manage raw MCP protocol details directly; it calls the higher-level helper functions exposed by that package.
