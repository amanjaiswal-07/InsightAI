# mcp/mcp_server.py
"""
Model Context Protocol (MCP) Server Implementation
Standard protocol for connecting LLMs to external systems
Based on: https://spec.modelcontextprotocol.io/specification/
"""
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from utils.logger import logger


@dataclass
class MCPTool:
    """MCP tool definition"""
    name: str
    description: str
    inputSchema: Dict[str, Any]  # JSON Schema
    handler: Callable  # Function to execute


@dataclass
class MCPResource:
    """MCP resource definition"""
    uri: str
    name: str
    description: str
    mimeType: str
    handler: Callable  # Function to fetch resource


class MCPServer:
    """
    MCP Server implementation
    Exposes tools and resources to MCP clients (like Claude)
    """

    def __init__(self, name: str, version: str):
        self.name = name
        self.version = version
        self.protocol_version = "2024-11-05"

        self.tools: Dict[str, MCPTool] = {}
        self.resources: Dict[str, MCPResource] = {}
        self.prompts: Dict[str, Dict[str, Any]] = {}

        logger.info(f"🔌 MCP Server initialized: {name} v{version}")

    def register_tool(self, tool: MCPTool):
        """Register a tool that Claude can call"""
        self.tools[tool.name] = tool
        logger.info(f"   🛠️  Tool registered: {tool.name}")

    def register_resource(self, resource: MCPResource):
        """Register a resource Claude can access"""
        self.resources[resource.uri] = resource
        logger.info(f"   📦 Resource registered: {resource.uri}")

    def register_prompt(self, name: str, description: str, arguments: List[Dict[str, Any]]):
        """Register a prompt template"""
        self.prompts[name] = {
            "name": name,
            "description": description,
            "arguments": arguments
        }
        logger.info(f"   💬 Prompt registered: {name}")

    async def handle_initialize(self, client_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle MCP initialization handshake
        Client → Server on connection
        """
        logger.info(f"🤝 MCP Client connected: {client_info.get('clientInfo', {}).get('name', 'unknown')}")

        return {
            "protocolVersion": self.protocol_version,
            "capabilities": {
                "tools": {"listChanged": True} if self.tools else {},
                "resources": {"subscribe": True, "listChanged": True} if self.resources else {},
                "prompts": {"listChanged": True} if self.prompts else {},
                "logging": {}
            },
            "serverInfo": {
                "name": self.name,
                "version": self.version
            }
        }

    async def handle_list_tools(self) -> Dict[str, Any]:
        """List all available tools"""
        tools_list = [
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.inputSchema
            }
            for tool in self.tools.values()
        ]

        return {"tools": tools_list}

    async def handle_call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool
        This is where Claude actually calls your functions
        """
        logger.info(f"🛠️  MCP Tool call: {name}")
        logger.info(f"   Arguments: {arguments}")

        tool = self.tools.get(name)

        if not tool:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Error: Tool '{name}' not found"
                    }
                ],
                "isError": True
            }

        try:
            # Execute the tool's handler
            result = await tool.handler(arguments)

            # Format response according to MCP spec
            if isinstance(result, dict) and "content" in result:
                return result
            else:
                # Wrap simple results
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": str(result)
                        }
                    ],
                    "isError": False
                }

        except Exception as e:
            logger.error(f"❌ Tool execution error: {e}")
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Error executing tool: {str(e)}"
                    }
                ],
                "isError": True
            }

    async def handle_list_resources(self) -> Dict[str, Any]:
        """List all available resources"""
        resources_list = [
            {
                "uri": resource.uri,
                "name": resource.name,
                "description": resource.description,
                "mimeType": resource.mimeType
            }
            for resource in self.resources.values()
        ]

        return {"resources": resources_list}

    async def handle_read_resource(self, uri: str) -> Dict[str, Any]:
        """Read a resource"""
        logger.info(f"📦 MCP Resource read: {uri}")

        resource = self.resources.get(uri)

        if not resource:
            return {
                "contents": [],
                "isError": True,
                "error": f"Resource not found: {uri}"
            }

        try:
            # Execute the resource's handler
            content = await resource.handler()

            return {
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": resource.mimeType,
                        "text": content if isinstance(content, str) else str(content)
                    }
                ]
            }

        except Exception as e:
            logger.error(f"❌ Resource read error: {e}")
            return {
                "contents": [],
                "isError": True,
                "error": str(e)
            }

    async def handle_list_prompts(self) -> Dict[str, Any]:
        """List all available prompts"""
        return {"prompts": list(self.prompts.values())}

    async def handle_get_prompt(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Get a prompt template"""
        prompt = self.prompts.get(name)

        if not prompt:
            return {"messages": [], "isError": True}

        return {
            "messages": [
                {
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": f"Prompt: {name}"
                    }
                }
            ]
        }

    def get_server_info(self) -> Dict[str, Any]:
        """Get server information for status endpoint"""
        return {
            "name": self.name,
            "version": self.version,
            "protocol_version": self.protocol_version,
            "capabilities": {
                "tools": len(self.tools),
                "resources": len(self.resources),
                "prompts": len(self.prompts)
            },
            "tools": list(self.tools.keys()),
            "resources": list(self.resources.keys()),
            "prompts": list(self.prompts.keys()),
            "status": "active",
            "timestamp": datetime.now().isoformat()
        }