# ABOUTME: Test suite for AutonomousFactory initialization and StrategicAgent communication
# ABOUTME: Validates factory lifecycle, agent integration, and message passing

"""Tests for AutonomousFactory initialization and StrategicAgent communication.

This module contains tests for the Measure phase, verifying:
1. AutonomousFactory initializes correctly
2. StrategicAgent initializes correctly
3. Factory can successfully communicate with StrategicAgent
4. Proper lifecycle management (init -> use -> shutdown)
"""

import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime

from ralph_orchestrator.factory import AutonomousFactory, FactoryConfig
from ralph_orchestrator.agents.strategic import StrategicAgent, AgentMessage, AgentState


class TestStrategicAgentInitialization(unittest.TestCase):
    """Test StrategicAgent initialization."""

    def test_agent_default_initialization(self):
        """Test agent initializes with default values."""
        agent = StrategicAgent()

        self.assertEqual(agent.name, "strategic_agent")
        self.assertEqual(agent.timeout, 300)
        self.assertEqual(agent.state, AgentState.UNINITIALIZED)
        self.assertFalse(agent.is_active)

    def test_agent_custom_initialization(self):
        """Test agent initializes with custom values."""
        agent = StrategicAgent(name="custom_agent", timeout=600)

        self.assertEqual(agent.name, "custom_agent")
        self.assertEqual(agent.timeout, 600)

    def test_agent_initialize_success(self):
        """Test agent initialization succeeds."""
        agent = StrategicAgent()

        result = agent.initialize()

        self.assertTrue(result)
        self.assertEqual(agent.state, AgentState.READY)
        self.assertTrue(agent.is_active)

    def test_agent_initialize_clears_previous_state(self):
        """Test initialization clears previous state."""
        agent = StrategicAgent()
        agent.initialize()

        # Add some tasks
        agent.add_task({"description": "Test task"})
        self.assertEqual(agent.task_count, 1)

        # Shutdown and reinitialize
        agent.shutdown()
        agent.initialize()

        self.assertEqual(agent.task_count, 0)
        self.assertEqual(agent.state, AgentState.READY)

    def test_agent_cannot_reinitialize_while_active(self):
        """Test agent cannot be reinitialized while already active."""
        agent = StrategicAgent()
        agent.initialize()

        result = agent.initialize()

        self.assertFalse(result)
        self.assertEqual(agent.state, AgentState.READY)

    def test_agent_get_status(self):
        """Test agent status reporting."""
        agent = StrategicAgent(name="test_agent")
        agent.initialize()

        status = agent.get_status()

        self.assertEqual(status["name"], "test_agent")
        self.assertEqual(status["state"], "ready")
        self.assertTrue(status["is_active"])
        self.assertEqual(status["pending_tasks"], 0)
        self.assertEqual(status["completed_tasks"], 0)


class TestStrategicAgentMessageProcessing(unittest.TestCase):
    """Test StrategicAgent message processing."""

    def setUp(self):
        """Set up test fixtures."""
        self.agent = StrategicAgent(name="test_agent")
        self.agent.initialize()

    def tearDown(self):
        """Clean up after tests."""
        self.agent.shutdown()

    def test_process_message_returns_response(self):
        """Test processing a message returns an AgentMessage."""
        response = self.agent.process_message("Test message")

        self.assertIsInstance(response, AgentMessage)
        self.assertIn("Test message", response.content)
        self.assertEqual(response.message_type, "response")

    def test_process_message_with_context(self):
        """Test processing a message with context."""
        context = {"task_id": "123", "priority": "high"}

        response = self.agent.process_message("Test message", context=context)

        self.assertIsNotNone(response)
        self.assertEqual(response.context.get("task_id"), "123")
        self.assertEqual(response.context.get("priority"), "high")

    def test_process_message_includes_metadata(self):
        """Test response includes expected metadata."""
        response = self.agent.process_message("Test message content")

        self.assertEqual(response.metadata["agent_name"], "test_agent")
        self.assertEqual(response.metadata["input_length"], len("Test message content"))

    def test_process_message_fails_when_not_ready(self):
        """Test message processing fails when agent not initialized."""
        agent = StrategicAgent()  # Not initialized

        response = agent.process_message("Test message")

        self.assertIsNone(response)

    def test_agent_message_to_dict(self):
        """Test AgentMessage converts to dictionary correctly."""
        message = AgentMessage(
            content="Test content",
            message_type="test",
            context={"key": "value"}
        )

        result = message.to_dict()

        self.assertEqual(result["content"], "Test content")
        self.assertEqual(result["message_type"], "test")
        self.assertEqual(result["context"]["key"], "value")
        self.assertIn("timestamp", result)


class TestStrategicAgentTaskManagement(unittest.TestCase):
    """Test StrategicAgent task management."""

    def setUp(self):
        """Set up test fixtures."""
        self.agent = StrategicAgent()
        self.agent.initialize()

    def tearDown(self):
        """Clean up after tests."""
        self.agent.shutdown()

    def test_add_task(self):
        """Test adding a task to the queue."""
        result = self.agent.add_task({"description": "Test task"})

        self.assertTrue(result)
        self.assertEqual(self.agent.task_count, 1)

    def test_add_task_requires_description(self):
        """Test adding task fails without description."""
        result = self.agent.add_task({"priority": "high"})

        self.assertFalse(result)
        self.assertEqual(self.agent.task_count, 0)

    def test_get_next_task(self):
        """Test getting the next task from queue."""
        self.agent.add_task({"description": "First task"})
        self.agent.add_task({"description": "Second task"})

        task = self.agent.get_next_task()

        self.assertEqual(task["description"], "First task")
        self.assertEqual(task["status"], "pending")

    def test_complete_task(self):
        """Test completing a task."""
        self.agent.add_task({"description": "Task to complete"})

        result = self.agent.complete_task()

        self.assertTrue(result)
        self.assertEqual(self.agent.task_count, 0)

    def test_complete_task_empty_queue(self):
        """Test completing task on empty queue."""
        result = self.agent.complete_task()

        self.assertFalse(result)


class TestAutonomousFactoryInitialization(unittest.TestCase):
    """Test AutonomousFactory initialization."""

    def test_factory_default_initialization(self):
        """Test factory initializes with default config."""
        factory = AutonomousFactory()

        self.assertFalse(factory.is_initialized)
        self.assertIsNone(factory.strategic_agent)
        self.assertEqual(len(factory.adapters), 0)

    def test_factory_custom_config(self):
        """Test factory initializes with custom config."""
        config = FactoryConfig(
            agent_timeout=600,
            max_retries=5,
            enable_strategic_planning=False
        )
        factory = AutonomousFactory(config=config)

        self.assertEqual(factory.config.agent_timeout, 600)
        self.assertEqual(factory.config.max_retries, 5)
        self.assertFalse(factory.config.enable_strategic_planning)

    def test_factory_initialize_creates_strategic_agent(self):
        """Test factory initialization creates strategic agent."""
        factory = AutonomousFactory()

        result = factory.initialize()

        self.assertTrue(result)
        self.assertTrue(factory.is_initialized)
        self.assertIsNotNone(factory.strategic_agent)
        self.assertTrue(factory.strategic_agent.is_active)

    def test_factory_initialize_is_idempotent(self):
        """Test calling initialize twice is safe."""
        factory = AutonomousFactory()
        factory.initialize()

        result = factory.initialize()

        self.assertTrue(result)
        self.assertTrue(factory.is_initialized)

    def test_factory_get_status(self):
        """Test factory status reporting."""
        factory = AutonomousFactory()
        factory.initialize()

        status = factory.get_status()

        self.assertTrue(status["initialized"])
        self.assertTrue(status["strategic_agent_active"])
        self.assertEqual(status["message_count"], 0)
        self.assertIn("config", status)


class TestAutonomousFactoryAdapterRegistration(unittest.TestCase):
    """Test AutonomousFactory adapter registration."""

    def setUp(self):
        """Set up test fixtures."""
        self.factory = AutonomousFactory()
        self.factory.initialize()

    def tearDown(self):
        """Clean up after tests."""
        self.factory.shutdown()

    def test_register_adapter(self):
        """Test registering an adapter."""
        mock_adapter = MagicMock()
        mock_adapter.name = "test_adapter"

        result = self.factory.register_adapter("test", mock_adapter)

        self.assertTrue(result)
        self.assertIn("test", self.factory.adapters)

    def test_register_duplicate_adapter_fails(self):
        """Test registering duplicate adapter name fails."""
        mock_adapter1 = MagicMock()
        mock_adapter2 = MagicMock()

        self.factory.register_adapter("test", mock_adapter1)
        result = self.factory.register_adapter("test", mock_adapter2)

        self.assertFalse(result)

    def test_adapters_returns_copy(self):
        """Test adapters property returns a copy."""
        mock_adapter = MagicMock()
        self.factory.register_adapter("test", mock_adapter)

        adapters = self.factory.adapters
        adapters["new"] = MagicMock()

        self.assertNotIn("new", self.factory.adapters)


class TestAutonomousFactoryStrategicAgentCommunication(unittest.TestCase):
    """Test communication between AutonomousFactory and StrategicAgent."""

    def setUp(self):
        """Set up test fixtures."""
        self.factory = AutonomousFactory()
        self.factory.initialize()

    def tearDown(self):
        """Clean up after tests."""
        self.factory.shutdown()

    def test_send_message_to_strategic_agent(self):
        """Test sending a message to the strategic agent."""
        response = self.factory.send_to_strategic_agent("Plan the implementation")

        self.assertIsInstance(response, AgentMessage)
        self.assertIn("Plan the implementation", response.content)

    def test_send_message_with_context(self):
        """Test sending a message with context."""
        context = {"project": "ralph-orchestrator", "phase": "measure"}

        response = self.factory.send_to_strategic_agent(
            "Analyze the requirements",
            context=context
        )

        self.assertIsNotNone(response)
        self.assertEqual(response.context.get("project"), "ralph-orchestrator")
        self.assertEqual(response.context.get("phase"), "measure")

    def test_message_history_tracking(self):
        """Test that message history is tracked."""
        self.factory.send_to_strategic_agent("First message")
        self.factory.send_to_strategic_agent("Second message")

        history = self.factory.get_message_history()

        self.assertEqual(len(history), 2)
        self.assertIn("First message", history[0].content)
        self.assertIn("Second message", history[1].content)

    def test_message_history_returns_copy(self):
        """Test message history returns a copy."""
        self.factory.send_to_strategic_agent("Test message")

        history = self.factory.get_message_history()
        history.clear()

        self.assertEqual(len(self.factory.get_message_history()), 1)

    def test_send_message_fails_when_not_initialized(self):
        """Test sending message fails when factory not initialized."""
        factory = AutonomousFactory()

        with self.assertRaises(RuntimeError) as context:
            factory.send_to_strategic_agent("Test message")

        self.assertIn("initialized", str(context.exception))

    def test_multiple_messages_maintain_context(self):
        """Test multiple messages maintain accumulated context."""
        self.factory.send_to_strategic_agent("First", context={"key1": "value1"})
        response = self.factory.send_to_strategic_agent("Second", context={"key2": "value2"})

        # Context should accumulate
        self.assertEqual(response.context.get("key1"), "value1")
        self.assertEqual(response.context.get("key2"), "value2")


class TestAutonomousFactoryShutdown(unittest.TestCase):
    """Test AutonomousFactory shutdown."""

    def test_factory_shutdown(self):
        """Test factory shutdown cleans up resources."""
        factory = AutonomousFactory()
        factory.initialize()

        mock_adapter = MagicMock()
        factory.register_adapter("test", mock_adapter)

        result = factory.shutdown()

        self.assertTrue(result)
        self.assertFalse(factory.is_initialized)
        self.assertIsNone(factory.strategic_agent)
        self.assertEqual(len(factory.adapters), 0)

    def test_factory_shutdown_is_idempotent(self):
        """Test calling shutdown twice is safe."""
        factory = AutonomousFactory()
        factory.initialize()

        factory.shutdown()
        result = factory.shutdown()

        self.assertTrue(result)

    def test_factory_can_reinitialize_after_shutdown(self):
        """Test factory can be reinitialized after shutdown."""
        factory = AutonomousFactory()
        factory.initialize()
        factory.send_to_strategic_agent("Test message")
        factory.shutdown()

        result = factory.initialize()

        self.assertTrue(result)
        self.assertTrue(factory.is_initialized)
        self.assertEqual(len(factory.get_message_history()), 0)


class TestIntegrationFactoryAndAgent(unittest.TestCase):
    """Integration tests for factory and agent working together."""

    def test_full_lifecycle(self):
        """Test complete lifecycle: init -> communicate -> shutdown."""
        # Initialize
        factory = AutonomousFactory(FactoryConfig(agent_timeout=120))
        self.assertTrue(factory.initialize())

        # Verify agent is ready
        self.assertTrue(factory.strategic_agent.is_active)
        self.assertEqual(factory.strategic_agent.state, AgentState.READY)

        # Communicate
        response = factory.send_to_strategic_agent(
            "Implement the feature",
            context={"feature": "user-auth"}
        )
        self.assertIsNotNone(response)

        # Verify status
        status = factory.get_status()
        self.assertTrue(status["initialized"])
        self.assertTrue(status["strategic_agent_active"])
        self.assertEqual(status["message_count"], 1)

        # Shutdown
        self.assertTrue(factory.shutdown())
        self.assertFalse(factory.is_initialized)

    def test_error_recovery(self):
        """Test factory handles agent errors gracefully."""
        factory = AutonomousFactory()
        factory.initialize()

        # Manually set agent to error state
        factory._strategic_agent._state = AgentState.ERROR

        # Should return None when agent in error state
        response = factory.send_to_strategic_agent("Test message")
        self.assertIsNone(response)

        # Factory should still be able to shutdown
        self.assertTrue(factory.shutdown())


if __name__ == "__main__":
    unittest.main()
