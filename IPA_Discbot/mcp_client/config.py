import os
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

MCPServerName = Literal["paas", "l2p"]

PAAS_SOLVE_TOOL = "paas_lama_first_solve"
L2P_DOMAIN_TOOL = "update_domain"
L2P_TASK_TOOL = "update_task"

_MCP_SERVER_URL_ENVS: dict[MCPServerName, str] = {
    "paas": "PAAS_MCP_URL",
    "l2p": "L2P_MCP_URL",
}

_MCP_SERVER_DEFAULT_URLS: dict[MCPServerName, str] = {
    "paas": "https://solver.planning.domains/mcp",
    "l2p": "http://127.0.0.1:8002/mcp",
}


def mcp_server_url(server: MCPServerName) -> str:
    env_name = _MCP_SERVER_URL_ENVS[server]
    default_url = _MCP_SERVER_DEFAULT_URLS[server]
    return os.getenv(env_name, default_url).strip() or default_url
