# MCP Client Package

This package contains the bot's MCP integration layer. Its job is to reach the configured MCP servers, call their tools, normalize their responses, and expose simple planning helpers to the rest of the bot.

## Files

- `config.py`: backend names, MCP tool names, default URLs, and environment-based URL resolution
- `manager.py`: persistent per-server MCP connections and shared connection manager functions
- `parsing.py`: helper functions for extracting text, dict payloads, and planner output from MCP responses
- `services.py`: higher-level operations such as solving PDDL, verifying servers, updating domain/task content, and listing tools
- `__init__.py`: package-level exports for the public MCP helper surface

## Backends

The client currently supports two MCP servers:

- `paas`: the planning-as-a-service backend used for solve requests
- `l2p`: the local `l2p-mcp` backend used for planning-edit helpers

Each backend resolves its URL from environment variables first, with a built-in fallback default.

For the `l2p` backend to work properly, the `l2p-mcp` service must be running separately as a Docker service. Repository:

- https://github.com/adam-neto/l2p_mcp

Current defaults:

- `PAAS_MCP_URL`: `https://solver.planning.domains/mcp`
- `L2P_MCP_URL`: `http://127.0.0.1:8002/mcp`

## Connection Model

The package keeps one shared MCP connection object per backend during bot runtime. Connections are established lazily on first use and then reused across later requests until shutdown.

This means:

- the bot does not need to connect to every backend at startup
- solve and validation requests reuse the same backend session once connected
- tool listing and edit helpers can share the same per-backend connection state

The explicit `connect_mcp_servers()` and `verify_*()` helpers are available for tests and manual debugging, even though the bot runtime doesn't use them.

## Public Operations

The main helpers exposed to the rest of the repo are:

- `connect_mcp_servers()`
- `close_mcp_servers()`
- `solve_pddl()`
- `validate_domain()`
- `validate_plan()`
- `validate_task()`
- `validate_plan_with_val()`
- `list_all_mcp_tools()`
- `refresh_mcp_tool_catalog()`
- `get_mcp_tool_catalog()`
- `update_domain_via_l2p()`
- `update_task_via_l2p()`
- `verify_mcp_server()`
- `verify_remote_mcp_server()`
- `verify_l2p_mcp_server()`

## Package Role In The Repo

The bot package calls into this package whenever a Discord interaction needs planning functionality. The MCP client package is intentionally separated so Discord-specific code stays in `bot/` while transport, tool calling, and response parsing stay here.
