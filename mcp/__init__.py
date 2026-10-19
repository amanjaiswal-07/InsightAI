# mcp/__init__.py
"""Minimal MCP-style tool server (JSON over the FastAPI routes in main.py)."""
from .mcp_server import MCPServer, MCPTool, MCPResource

__all__ = ["MCPServer", "MCPTool", "MCPResource"]
