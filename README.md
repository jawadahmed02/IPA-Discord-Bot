# Demonstration
[![User demo video](https://cdn.loom.com/sessions/thumbnails/7c9f8f98e8f44776b55249da51d1f004-64bc52969c2346b5-full-play.gif#t=0.1)](https://www.loom.com/share/7c9f8f98e8f44776b55249da51d1f004)

# IPA_Discbot

`IPA_Discbot` is a Discord-based interactive planning assistant. It combines a conversational bot, persistent user context, model selection through the `llm` Python package, and MCP-backed planning services so users can move between natural-language requests and planning tools inside Discord.

## Repository Layout

The repo is organized around two main internal packages plus a small set of top-level project files:

- `IPA_Discbot/bot/` contains the Discord bot runtime, command handlers, conversation flow, and persistence logic.
- `IPA_Discbot/mcp_client/` contains the MCP integration layer that talks to the planning backends.
- `requirements.txt`, `Dockerfile`, and `docker-compose.yml` hold installation and deployment setup.

## High-Level Architecture

At runtime, the flow is:

1. The Discord bot starts from `IPA_Discbot.bot`.
2. Environment configuration is loaded and the bot instance is created.
3. Discord messages are classified into chat, workflow, helper, or command paths.
4. Discord messages are routed either into command handlers, planning helpers, member/thread actions, or normal LLM chat.
5. Conversation history and per-user model/provider settings are stored in SQLite.
6. Planning requests are sent through the MCP client layer, and results are returned back into Discord.

## Bot Layer

The `bot/` package is the user-facing layer. It handles Discord interaction, conversation state, user settings, and the planning workflow that users see in chat.

Its main capabilities are:

- normal conversational replies with persisted context
- per-user model selection
- per-user provider key storage
- planning requests from Discord attachments with `!plan`
- natural-language `!plan`, `!domain`, and `!problem` flows that go through the local `l2p` MCP server before solving
- artifact inspection and revision with `!show`, `!edit`, and `!undo`
- shared channel collaboration with `!collab` for shared chat context and shared planning artifacts
- PDDL syntax checking with `!validate_domain`, `!validate_task`, and `!validate_plan`
- Validate fit between a domain/problem pair and a plan with `!validate`
- thread creation, member lookup, and thread-add helper flows
- session saving and provider-key sharing controls
- MCP tool listing across both configured servers with `!tools` and `!paastools`

In practice, this layer interprets Discord messages, decides when to call the planning services, tracks the current working domain/problem/plan for a user or shared channel, and formats results back into Discord replies and files.

## Setup

Install dependencies from the repo root with:

```bash
python3 -m pip install -r requirements.txt
```

Before running the bot, you also need the `l2p-mcp` service running as a Docker service. The bot depends on that MCP backend for planning-edit flows and related `l2p` tooling:

- `l2p-mcp`: https://github.com/adam-neto/l2p_mcp

Set up a `.env` file. You can start from `.env.example`.

Required values:

- `DISCORD_TOKEN`
- `BOT_MASTER_KEY`
- `DISCORD_GUILD_ID`

Optional runtime settings:

- `OPENAI_MODEL` defaults to `gpt-4.1`
- `DB_PATH` defaults to `bot.db`

Provider API keys are supplied in Discord with `/setkey` and stored encrypted in SQLite, rather than being loaded from `.env`.

Optional MCP endpoint overrides for running the bot directly on your host:

- `PAAS_MCP_URL` defaults to `https://solver.planning.domains/mcp`
- `L2P_MCP_URL` defaults to `http://127.0.0.1:8002/mcp`

## MCP Layer

The `mcp_client/` package is the planning-service adapter for the bot. It hides MCP transport details, knows how to reach the configured planning backends, and turns backend responses into simpler values the bot can use.

The current MCP-based backends are:

- `paas` for planning solve requests
- `l2p` for local planning-editing tools

Its main responsibilities are:

- call solver and validation tools on the remote planning backends
- expose higher-level operations like solve, validate, and planning-edit helpers
- normalize planner and validation payloads before they reach the bot layer
- keep backend-specific tool names and endpoint resolution in one place

## Running The Bot

Make sure the `l2p-mcp` Docker service is already running and reachable at the `L2P_MCP_URL` you configured before starting the bot.

Start the bot from the repo root with:

```bash
python3 -m IPA_Discbot.bot
```

Startup initializes the database, loads environment variables, constructs the Discord bot, and registers handlers before serving requests. Planning features connect to the MCP backends when they need them.

## Docker

This repo includes a small `Dockerfile` and `docker-compose.yml` for running the Discord bot as a long-lived service.

The container setup is configured to:

- run the Discord bot as a background service
- restart automatically unless you stop it explicitly
- persist the SQLite database in a Docker volume
- load runtime secrets and overrides from `.env`
- default `DB_PATH` to `/data/bot.db`
- set `PAAS_MCP_URL` to `https://solver.planning.domains/mcp`
- set `L2P_MCP_URL` to `http://host.docker.internal:8002/mcp` so the bot container can reach a separately running `l2p-mcp` service on the host

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
