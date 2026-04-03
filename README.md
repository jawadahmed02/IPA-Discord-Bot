# Demonstration
[![User demo video](https://cdn.loom.com/sessions/thumbnails/7c9f8f98e8f44776b55249da51d1f004-64bc52969c2346b5-full-play.gif#t=0.1)](https://www.loom.com/share/7c9f8f98e8f44776b55249da51d1f004)

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

Before running the bot, you also need the `l2p-mcp` service running as a Docker service. The bot depends on that MCP backend for planning-edit flows and related `l2p` tooling:

- `l2p-mcp`: https://github.com/adam-neto/l2p_mcp

Set up a `.env` file. You can start from `.env.example`.

Required values:

- `DISCORD_TOKEN`
- `BOT_MASTER_KEY`
- `DISCORD_GUILD_ID`

Optional provider keys:

- `OPENAI_API_KEY`
- `LLM_GEMINI_KEY`
- `ANTHROPIC_API_KEY`

Optional runtime settings:

- `OPENAI_MODEL` defaults to `gpt-4.1`
- `DB_PATH` defaults to `bot.db`

Optional MCP endpoint overrides for running the bot directly on your host:

- `PAAS_MCP_URL` defaults to `https://solver.planning.domains/mcp`
- `L2P_MCP_URL` defaults to `http://127.0.0.1:8002/mcp`

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

Make sure the `l2p-mcp` Docker service is already running and reachable at the `L2P_MCP_URL` you configured before starting the bot.

Start the bot from the repo root with:

```bash
python3 -m IPA_Discbot.bot
```

Startup initializes the database, loads environment variables, constructs the Discord bot, registers handlers, and connects to both MCP servers before serving requests.

## Docker

This repo includes a small `Dockerfile` and `docker-compose.yml` for running the Discord bot as a long-lived service.

The container setup is configured to:

- run the Discord bot as a background service
- restart automatically unless you stop it explicitly
- persist the SQLite database in a Docker volume
- load runtime secrets and overrides from `.env`
- set image-level defaults for `OPENAI_MODEL`, `DB_PATH`, `PAAS_MCP_URL`, and `L2P_MCP_URL`
- default `DB_PATH` to `/data/bot.db`
- default `PAAS_MCP_URL` to `https://solver.planning.domains/mcp`
- default `L2P_MCP_URL` to `http://host.docker.internal:8002/mcp` so the bot container can reach a separately running `l2p-mcp` service on the host

Once the required MCP backends are reachable, start the bot from the repo root with:

```bash
docker compose up -d --build
```

Helpful commands:

- `docker compose logs -f`
- `docker compose ps`
- `docker compose restart`
- `docker compose down`

By default, bot data is stored in the named Docker volume `ipa-discbot-data`. If you prefer a host-mounted database path instead, update `DB_PATH` and the Compose volume mapping together.
