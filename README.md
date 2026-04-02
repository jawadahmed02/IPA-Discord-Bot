# Demonstration
<div style="position: relative; padding-bottom: 49.36014625228518%; height: 0;"><iframe src="https://www.loom.com/embed/7c9f8f98e8f44776b55249da51d1f004" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe></div>

# IPA_Discbot

`IPA_Discbot` is a Discord-based interactive planning assistant. It combines a conversational bot, persistent user context, model selection through the `llm` Python package, and MCP-backed planning services so users can move between natural-language requests and planning tools inside Discord.

## Repository Layout

The repo is organized around two main internal packages:

- `IPA_Discbot/bot/` contains the Discord bot runtime, command handlers, conversation flow, and persistence logic.
- `IPA_Discbot/mcp_client/` contains the MCP integration layer that talks to the planning backends.

## High-Level Architecture

At runtime, the flow is:

1. The Discord bot starts from `IPA_Discbot.bot`.
2. Environment configuration is loaded and the bot instance is created.
3. On startup, the bot opens live MCP connections to both configured backends.
4. Discord messages are routed either into command handlers, planning helpers, member/thread actions, or normal LLM chat.
5. Conversation history and per-user model/provider settings are stored in SQLite.
6. Planning requests are sent through the MCP client layer, and results are returned back into Discord.

## Bot Layer

The `bot/` package is split by responsibility:

- `config.py` defines shared runtime configuration and the bot instance.
- `storage.py` handles SQLite persistence, encrypted provider keys, and message history.
- `llm_helpers.py` wraps model prompting and lightweight LLM classification helpers.
- `services.py` contains Discord commands, event handlers, and message-routing behavior.
- `__init__.py` exposes the package entrypoint via `run()`.
- `__main__.py` makes the package runnable with `python3 -m IPA_Discbot.bot`.

The bot currently supports:

- normal conversational replies with persisted context
- per-user model selection
- per-user provider key storage
- planning requests from Discord attachments with `!plan`
- natural-language `!plan`, `!domain`, and `!problem` flows that go through the local `l2p` MCP server before solving
- artifact inspection and revision with `!show`, `!edit`, and `!undo`
- plan, domain, task, and VAL-based validation flows with `!validate`, `!validate_domain`, `!validate_task`, and `!autovalidate`
- thread creation, member lookup, and thread-add helper flows
- session saving and provider-key sharing controls
- MCP tool listing across both configured servers with `!tools` and `!paastools`

## Setup

Install dependencies from the repo root with:

```bash
python3 -m pip install -r IPA_Discbot/requirements.txt
```

Set up a `.env` file with at least:

- `DISCORD_TOKEN`
- `BOT_MASTER_KEY`
- `PAAS_MCP_URL`
- `L2P_MCP_URL`

Optional provider keys:

- `OPENAI_API_KEY`
- `LLM_GEMINI_KEY`
- `ANTHROPIC_API_KEY`

Optional model override:

- `OPENAI_MODEL` defaults to `gpt-4.1`

## MCP Layer

The `mcp_client/` package is the interface between the bot and the planning services. It keeps one live connection to each supported backend and exposes simple helpers that the bot can call.

The MCP package is split into:

- `config.py` for backend names, tool names, and endpoint resolution
- `manager.py` for persistent MCP session lifecycle
- `parsing.py` for normalizing MCP responses
- `services.py` for higher-level bot-facing MCP operations

The current backends are:

- `paas` for planning solve requests
- `l2p` for local planning-editing tools

## Running The Bot

Start the bot from the repo root with:

```bash
python3 -m IPA_Discbot.bot
```

Startup initializes the database, loads environment variables, constructs the Discord bot, registers handlers, and connects to both MCP servers before serving requests.
