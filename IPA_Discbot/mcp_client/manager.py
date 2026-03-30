import asyncio
import contextlib
from typing import Any

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

from .config import MCPServerName, mcp_server_url
from .parsing import tool_payload


class LiveMCPServerConnection:
    def __init__(self, server: MCPServerName):
        self.server = server
        self._lock = asyncio.Lock()
        self._transport_cm = None
        self._session_cm = None
        self._session: ClientSession | None = None

    async def connect(self) -> ClientSession:
        async with self._lock:
            return await self._connect_locked()

    async def _connect_locked(self) -> ClientSession:
        if self._session is not None:
            return self._session

        transport_cm = None
        session_cm = None
        try:
            transport_cm = streamable_http_client(mcp_server_url(self.server))
            read, write, _ = await transport_cm.__aenter__()
            session_cm = ClientSession(read, write)
            session = await session_cm.__aenter__()
            await session.initialize()
        except Exception:
            with contextlib.suppress(BaseException):
                await session_cm.__aexit__(None, None, None)
            with contextlib.suppress(BaseException):
                await transport_cm.__aexit__(None, None, None)
            raise

        self._transport_cm = transport_cm
        self._session_cm = session_cm
        self._session = session
        return session

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        async with self._lock:
            session = await self._connect_locked()
            result = await session.call_tool(tool_name, arguments)
            return tool_payload(result)

    async def list_tools(self) -> list[dict[str, Any]]:
        async with self._lock:
            session = await self._connect_locked()
            tools = await session.list_tools()
            return [
                {
                    "name": tool.name,
                    "description": (getattr(tool, "description", None) or "").strip(),
                    "input_schema": getattr(tool, "inputSchema", None)
                    or getattr(tool, "input_schema", None),
                    "output_schema": getattr(tool, "outputSchema", None)
                    or getattr(tool, "output_schema", None),
                }
                for tool in getattr(tools, "tools", [])
            ]

    async def close(self) -> None:
        async with self._lock:
            transport_cm = self._transport_cm
            session_cm = self._session_cm
            self._transport_cm = None
            self._session_cm = None
            self._session = None
            if session_cm is not None:
                with contextlib.suppress(BaseException):
                    await session_cm.__aexit__(None, None, None)
            if transport_cm is not None:
                with contextlib.suppress(BaseException):
                    await transport_cm.__aexit__(None, None, None)


class MCPConnectionManager:
    def __init__(self):
        self._servers = {
            "paas": LiveMCPServerConnection("paas"),
            "l2p": LiveMCPServerConnection("l2p"),
        }

    async def connect_all(self) -> None:
        await self._servers["paas"].connect()
        await self._servers["l2p"].connect()

    async def close_all(self) -> None:
        await self._servers["paas"].close()
        await self._servers["l2p"].close()

    async def call_tool(
        self,
        server: MCPServerName,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        return await self._servers[server].call_tool(tool_name, arguments)

    async def list_tools(self, server: MCPServerName) -> list[dict[str, Any]]:
        return await self._servers[server].list_tools()


_MCP_CONNECTIONS = MCPConnectionManager()


async def connect_mcp_servers() -> None:
    await _MCP_CONNECTIONS.connect_all()


async def close_mcp_servers() -> None:
    await _MCP_CONNECTIONS.close_all()


async def call_mcp_tool(
    server: MCPServerName,
    tool_name: str,
    arguments: dict[str, Any],
) -> Any:
    return await _MCP_CONNECTIONS.call_tool(server, tool_name, arguments)


async def list_mcp_tools(server: MCPServerName) -> list[dict[str, Any]]:
    return await _MCP_CONNECTIONS.list_tools(server)
