# protocols/a2a.py
"""
Agent-to-Agent (A2A) Protocol Implementation
Enables autonomous agent discovery and communication
"""
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
import asyncio
from utils.logger import logger


class MessageType(Enum):
    """A2A message types"""
    QUERY = "query"
    RESPONSE = "response"
    DELEGATE = "delegate"
    CAPABILITY_REQUEST = "capability_request"
    CAPABILITY_RESPONSE = "capability_response"
    ERROR = "error"


@dataclass
class AgentCapability:
    """Agent capability definition"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]


@dataclass
class A2AMessage:
    """A2A message envelope"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    from_agent: str = ""
    to_agent: str = ""
    message_type: MessageType = MessageType.QUERY
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    correlation_id: Optional[str] = None  # For tracking request-response pairs


class A2AAgent:
    """Base class for A2A-enabled agents"""

    def __init__(self, agent_id: str, name: str, description: str):
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.capabilities: Dict[str, AgentCapability] = {}
        self.message_handlers: Dict[MessageType, Callable] = {}
        self.connected_agents: Dict[str, Dict[str, Any]] = {}

        # Register default handlers
        self._register_default_handlers()

        logger.info(f"🤖 A2A Agent initialized: {agent_id}")

    def _register_default_handlers(self):
        """Register default message handlers"""
        self.message_handlers[MessageType.CAPABILITY_REQUEST] = self._handle_capability_request
        self.message_handlers[MessageType.ERROR] = self._handle_error

    def register_capability(self, capability: AgentCapability):
        """Register a capability this agent can perform"""
        self.capabilities[capability.name] = capability
        logger.info(f"   ✅ Capability registered: {capability.name}")

    def register_handler(self, message_type: MessageType, handler: Callable):
        """Register custom message handler"""
        self.message_handlers[message_type] = handler

    async def send_message(self, to_agent: str, message_type: MessageType,
                          payload: Dict[str, Any], correlation_id: Optional[str] = None) -> A2AMessage:
        """Send A2A message to another agent"""
        message = A2AMessage(
            from_agent=self.agent_id,
            to_agent=to_agent,
            message_type=message_type,
            payload=payload,
            correlation_id=correlation_id
        )

        logger.info(f"📤 A2A: {self.agent_id} → {to_agent} ({message_type.value})")

        return message

    async def receive_message(self, message: A2AMessage) -> A2AMessage:
        """Receive and process A2A message"""
        logger.info(f"📥 A2A: {message.from_agent} → {self.agent_id} ({message.message_type.value})")

        handler = self.message_handlers.get(message.message_type)

        if handler:
            try:
                response_payload = await handler(message)

                return A2AMessage(
                    from_agent=self.agent_id,
                    to_agent=message.from_agent,
                    message_type=MessageType.RESPONSE,
                    payload=response_payload,
                    correlation_id=message.id
                )
            except Exception as e:
                logger.error(f"❌ Handler error: {e}")
                return self._create_error_message(message, str(e))
        else:
            return self._create_error_message(message, f"No handler for {message.message_type}")

    async def _handle_capability_request(self, message: A2AMessage) -> Dict[str, Any]:
        """Handle capability discovery request"""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "capabilities": [
                {
                    "name": cap.name,
                    "description": cap.description,
                    "input_schema": cap.input_schema,
                    "output_schema": cap.output_schema
                }
                for cap in self.capabilities.values()
            ]
        }

    async def _handle_error(self, message: A2AMessage) -> Dict[str, Any]:
        """Handle error messages"""
        logger.error(f"❌ A2A Error from {message.from_agent}: {message.payload.get('error')}")
        return {"acknowledged": True}

    def _create_error_message(self, original_message: A2AMessage, error: str) -> A2AMessage:
        """Create error response message"""
        return A2AMessage(
            from_agent=self.agent_id,
            to_agent=original_message.from_agent,
            message_type=MessageType.ERROR,
            payload={"error": error},
            correlation_id=original_message.id
        )

    def discover_agent(self, agent_id: str) -> Dict[str, Any]:
        """Discover another agent's capabilities"""
        return self.connected_agents.get(agent_id, {})

    def register_peer(self, agent_id: str, manifest: Dict[str, Any]):
        """Register a peer agent"""
        self.connected_agents[agent_id] = manifest
        logger.info(f"   🔗 Peer registered: {agent_id}")

    def get_manifest(self) -> Dict[str, Any]:
        """Get agent manifest for discovery"""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "capabilities": [cap.name for cap in self.capabilities.values()],
            "endpoints": {
                "message": f"/a2a/{self.agent_id}/message",
                "capabilities": f"/a2a/{self.agent_id}/capabilities"
            }
        }


class A2ARegistry:
    """Central registry for agent discovery"""

    def __init__(self):
        self.agents: Dict[str, A2AAgent] = {}
        logger.info("📋 A2A Registry initialized")

    def register_agent(self, agent: A2AAgent):
        """Register agent in registry"""
        self.agents[agent.agent_id] = agent
        logger.info(f"   ✅ Agent registered in registry: {agent.agent_id}")

        # Notify all other agents about new peer
        for other_agent in self.agents.values():
            if other_agent.agent_id != agent.agent_id:
                other_agent.register_peer(agent.agent_id, agent.get_manifest())
                agent.register_peer(other_agent.agent_id, other_agent.get_manifest())

    def get_agent(self, agent_id: str) -> Optional[A2AAgent]:
        """Get agent by ID"""
        return self.agents.get(agent_id)

    def discover_agents(self, capability: Optional[str] = None) -> List[Dict[str, Any]]:
        """Discover agents, optionally filtered by capability"""
        agents = []

        for agent in self.agents.values():
            if capability:
                if capability in agent.capabilities:
                    agents.append(agent.get_manifest())
            else:
                agents.append(agent.get_manifest())

        return agents

    async def route_message(self, message: A2AMessage) -> A2AMessage:
        """Route message to target agent"""
        target_agent = self.agents.get(message.to_agent)

        if not target_agent:
            return A2AMessage(
                from_agent="registry",
                to_agent=message.from_agent,
                message_type=MessageType.ERROR,
                payload={"error": f"Agent not found: {message.to_agent}"},
                correlation_id=message.id
            )

        return await target_agent.receive_message(message)