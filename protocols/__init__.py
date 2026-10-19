# protocols/__init__.py
"""
Lightweight agent-to-agent (A2A) messaging used by the orchestrator.
"""

from .a2a import (
    A2AAgent,
    A2AMessage,
    AgentCapability,
    MessageType,
    A2ARegistry
)

__all__ = [
    'A2AAgent',
    'A2AMessage',
    'AgentCapability',
    'MessageType',
    'A2ARegistry'
]