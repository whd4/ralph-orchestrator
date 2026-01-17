# ABOUTME: Factory class for creating and managing autonomous orchestration instances
# ABOUTME: Provides centralized initialization of orchestrators, adapters, and strategic agents

"""Autonomous Factory for Ralph Orchestrator.

This module provides the AutonomousFactory class which serves as the central
point for creating and managing orchestrator instances with their associated
strategic agents and adapters.
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

from .adapters.base import ToolAdapter, ToolResponse
from .agents.strategic import StrategicAgent, AgentMessage


@dataclass
class FactoryConfig:
    """Configuration for the AutonomousFactory."""

    agent_timeout: int = 300
    max_retries: int = 3
    enable_strategic_planning: bool = True
    adapter_configs: Dict[str, Any] = field(default_factory=dict)


class AutonomousFactory:
    """Factory for creating and managing autonomous orchestration components.

    The AutonomousFactory provides a centralized way to:
    - Initialize and configure adapters
    - Create and manage StrategicAgent instances
    - Coordinate communication between components
    - Handle lifecycle management of orchestration components

    Example:
        factory = AutonomousFactory()
        factory.initialize()

        # Send a message to the strategic agent
        response = factory.send_to_strategic_agent("Plan the implementation")

        # Shutdown when done
        factory.shutdown()
    """

    def __init__(self, config: Optional[FactoryConfig] = None):
        """Initialize the AutonomousFactory.

        Args:
            config: Optional factory configuration. If not provided,
                   defaults will be used.
        """
        self.config = config or FactoryConfig()
        self._initialized = False
        self._strategic_agent: Optional[StrategicAgent] = None
        self._adapters: Dict[str, ToolAdapter] = {}
        self._message_history: List[AgentMessage] = []

    @property
    def is_initialized(self) -> bool:
        """Check if the factory has been initialized."""
        return self._initialized

    @property
    def strategic_agent(self) -> Optional[StrategicAgent]:
        """Get the current strategic agent instance."""
        return self._strategic_agent

    @property
    def adapters(self) -> Dict[str, ToolAdapter]:
        """Get all registered adapters."""
        return self._adapters.copy()

    def initialize(self) -> bool:
        """Initialize the factory and all components.

        This sets up the strategic agent and any configured adapters.
        Must be called before using the factory for communication.

        Returns:
            True if initialization was successful, False otherwise.
        """
        if self._initialized:
            return True

        try:
            # Clear any previous message history
            self._message_history.clear()

            # Create the strategic agent
            self._strategic_agent = StrategicAgent(
                name="primary_strategic_agent",
                timeout=self.config.agent_timeout
            )

            # Initialize the strategic agent
            if not self._strategic_agent.initialize():
                return False

            self._initialized = True
            return True

        except Exception:
            self._initialized = False
            return False

    def register_adapter(self, name: str, adapter: ToolAdapter) -> bool:
        """Register a tool adapter with the factory.

        Args:
            name: Unique name for the adapter
            adapter: The ToolAdapter instance to register

        Returns:
            True if registration was successful, False if name already exists
        """
        if name in self._adapters:
            return False

        self._adapters[name] = adapter
        return True

    def send_to_strategic_agent(self, message: str, context: Optional[Dict[str, Any]] = None) -> Optional[AgentMessage]:
        """Send a message to the strategic agent.

        Args:
            message: The message content to send
            context: Optional context dictionary for the message

        Returns:
            AgentMessage response from the strategic agent, or None if failed

        Raises:
            RuntimeError: If the factory has not been initialized
        """
        if not self._initialized or self._strategic_agent is None:
            raise RuntimeError("Factory must be initialized before sending messages")

        response = self._strategic_agent.process_message(message, context)

        if response:
            self._message_history.append(response)

        return response

    def get_message_history(self) -> List[AgentMessage]:
        """Get the history of messages exchanged with the strategic agent.

        Returns:
            List of AgentMessage objects
        """
        return self._message_history.copy()

    def shutdown(self) -> bool:
        """Shutdown the factory and all components.

        Returns:
            True if shutdown was successful, False otherwise
        """
        if not self._initialized:
            return True

        try:
            if self._strategic_agent:
                self._strategic_agent.shutdown()
                self._strategic_agent = None

            self._adapters.clear()
            self._initialized = False
            return True

        except Exception:
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the factory.

        Returns:
            Dictionary containing factory status information
        """
        return {
            "initialized": self._initialized,
            "strategic_agent_active": self._strategic_agent is not None and self._strategic_agent.is_active,
            "registered_adapters": list(self._adapters.keys()),
            "message_count": len(self._message_history),
            "config": {
                "agent_timeout": self.config.agent_timeout,
                "max_retries": self.config.max_retries,
                "enable_strategic_planning": self.config.enable_strategic_planning,
            }
        }
