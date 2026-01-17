# ABOUTME: Strategic agent for high-level planning and coordination
# ABOUTME: Handles task decomposition, prioritization, and strategic decision-making

"""Strategic Agent for Ralph Orchestrator.

This module provides the StrategicAgent class which handles high-level
planning, task decomposition, and strategic coordination within the
orchestration system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List


class AgentState(Enum):
    """Possible states for the StrategicAgent."""

    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class AgentMessage:
    """A message exchanged with the StrategicAgent."""

    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    message_type: str = "response"
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary representation."""
        return {
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "message_type": self.message_type,
            "context": self.context,
            "metadata": self.metadata,
        }


class StrategicAgent:
    """Agent responsible for strategic planning and coordination.

    The StrategicAgent handles:
    - High-level task decomposition
    - Priority assignment for subtasks
    - Strategic decision-making during orchestration
    - Coordination between multiple adapters

    Example:
        agent = StrategicAgent(name="planner")
        agent.initialize()

        response = agent.process_message("Plan the feature implementation")
        print(response.content)

        agent.shutdown()
    """

    def __init__(self, name: str = "strategic_agent", timeout: int = 300):
        """Initialize the StrategicAgent.

        Args:
            name: Unique name for this agent instance
            timeout: Default timeout for processing operations (seconds)
        """
        self.name = name
        self.timeout = timeout
        self._state = AgentState.UNINITIALIZED
        self._context: Dict[str, Any] = {}
        self._task_queue: List[Dict[str, Any]] = []
        self._completed_tasks: List[Dict[str, Any]] = []

    @property
    def state(self) -> AgentState:
        """Get the current agent state."""
        return self._state

    @property
    def is_active(self) -> bool:
        """Check if the agent is in an active state."""
        return self._state in (AgentState.READY, AgentState.PROCESSING)

    @property
    def task_count(self) -> int:
        """Get the number of pending tasks."""
        return len(self._task_queue)

    def initialize(self) -> bool:
        """Initialize the agent and prepare for processing.

        Returns:
            True if initialization was successful, False otherwise
        """
        if self._state not in (AgentState.UNINITIALIZED, AgentState.SHUTDOWN):
            return False

        try:
            self._state = AgentState.INITIALIZING

            # Clear any previous state
            self._context.clear()
            self._task_queue.clear()
            self._completed_tasks.clear()

            self._state = AgentState.READY
            return True

        except Exception:
            self._state = AgentState.ERROR
            return False

    def process_message(self, message: str, context: Optional[Dict[str, Any]] = None) -> Optional[AgentMessage]:
        """Process an incoming message and generate a response.

        Args:
            message: The message content to process
            context: Optional context dictionary for the message

        Returns:
            AgentMessage response, or None if processing failed
        """
        if self._state != AgentState.READY:
            return None

        try:
            self._state = AgentState.PROCESSING

            # Update context with provided context
            if context:
                self._context.update(context)

            # Generate response (skeleton implementation)
            response_content = self._generate_response(message)

            response = AgentMessage(
                content=response_content,
                message_type="response",
                context=self._context.copy(),
                metadata={
                    "agent_name": self.name,
                    "input_length": len(message),
                }
            )

            self._state = AgentState.READY
            return response

        except Exception:
            self._state = AgentState.ERROR
            return None

    def _generate_response(self, message: str) -> str:
        """Generate a response to the given message.

        This is a skeleton implementation that provides basic acknowledgment.
        Production implementations would integrate with actual AI models.

        Args:
            message: The input message to respond to

        Returns:
            Generated response string
        """
        # Skeleton response - acknowledges receipt and provides basic info
        return f"Strategic analysis complete for: {message[:50]}{'...' if len(message) > 50 else ''}"

    def add_task(self, task: Dict[str, Any]) -> bool:
        """Add a task to the agent's queue.

        Args:
            task: Task dictionary with at least 'description' key

        Returns:
            True if task was added successfully
        """
        if "description" not in task:
            return False

        task_with_meta = {
            **task,
            "added_at": datetime.now().isoformat(),
            "status": "pending",
        }
        self._task_queue.append(task_with_meta)
        return True

    def get_next_task(self) -> Optional[Dict[str, Any]]:
        """Get the next task from the queue.

        Returns:
            Next task dictionary, or None if queue is empty
        """
        if not self._task_queue:
            return None

        return self._task_queue[0]

    def complete_task(self, task_id: Optional[str] = None) -> bool:
        """Mark the current/specified task as complete.

        Args:
            task_id: Optional specific task ID to complete.
                    If not provided, completes the first task in queue.

        Returns:
            True if a task was completed
        """
        if not self._task_queue:
            return False

        task = self._task_queue.pop(0)
        task["status"] = "completed"
        task["completed_at"] = datetime.now().isoformat()
        self._completed_tasks.append(task)
        return True

    def shutdown(self) -> bool:
        """Shutdown the agent gracefully.

        Returns:
            True if shutdown was successful
        """
        try:
            self._state = AgentState.SHUTDOWN
            self._context.clear()
            return True
        except Exception:
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the agent.

        Returns:
            Dictionary containing agent status information
        """
        return {
            "name": self.name,
            "state": self._state.value,
            "is_active": self.is_active,
            "timeout": self.timeout,
            "pending_tasks": len(self._task_queue),
            "completed_tasks": len(self._completed_tasks),
        }
