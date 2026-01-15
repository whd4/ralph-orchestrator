# ABOUTME: Tests for Strategic Agent (CEO-level decision making)
# ABOUTME: Verifies task management and strategic decisions

"""Tests for the Strategic Agent."""

import pytest
from pathlib import Path
import tempfile
import shutil

from ralph_orchestrator.strategic_agent import (
    StrategicAgent,
    Task,
    Priority,
    RiskLevel,
    StrategicDecision,
    CEOPrinciples,
    get_strategic_agent,
)


class TestStrategicAgent:
    """Tests for StrategicAgent class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp = tempfile.mkdtemp()
        yield Path(temp)
        shutil.rmtree(temp, ignore_errors=True)

    @pytest.fixture
    def agent(self, temp_dir):
        """Create agent with temp memory path."""
        return StrategicAgent(
            name="TestCEO",
            token_budget=100000,
            risk_tolerance=0.5,
            memory_path=temp_dir / "strategic_memory.json"
        )

    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent.name == "TestCEO"
        assert agent.token_budget == 100000
        assert agent.tokens_used == 0
        assert agent.risk_tolerance == 0.5

    def test_add_task(self, agent):
        """Test adding a task."""
        task = Task(
            id="task_1",
            description="Implement user authentication",
            priority=Priority.HIGH,
            estimated_effort=5
        )

        agent.add_task(task)

        assert len(agent.task_queue) == 1
        assert agent.task_queue[0].id == "task_1"

    def test_add_tasks_from_prompt(self, agent):
        """Test extracting tasks from prompt."""
        prompt = """
        - [ ] Implement login functionality
        - [ ] Add password reset
        - [ ] Create user dashboard
        """

        tasks = agent.add_tasks_from_prompt(prompt)

        assert len(tasks) == 3
        assert all(isinstance(t, Task) for t in tasks)

    def test_task_priority_estimation(self, agent):
        """Test priority estimation from description."""
        prompt = """
        - [ ] Critical security fix needed immediately
        - [ ] Optional nice-to-have feature
        - [ ] Standard implementation task
        """

        tasks = agent.add_tasks_from_prompt(prompt)

        priorities = [t.priority for t in tasks]
        assert Priority.CRITICAL in priorities
        assert Priority.MEDIUM in priorities or Priority.LOW in priorities

    def test_get_next_task(self, agent):
        """Test getting next task."""
        task1 = Task(id="t1", description="Low priority", priority=Priority.LOW)
        task2 = Task(id="t2", description="High priority", priority=Priority.HIGH)

        agent.add_task(task1)
        agent.add_task(task2)

        next_task = agent.get_next_task()

        assert next_task.priority == Priority.HIGH

    def test_make_decision(self, agent):
        """Test making a strategic decision."""
        task = Task(
            id="task_1",
            description="Implement REST API endpoints",
            priority=Priority.MEDIUM,
            estimated_effort=5
        )

        decision = agent.make_decision(task)

        assert isinstance(decision, StrategicDecision)
        assert decision.task == task
        assert 0 <= decision.confidence <= 1
        assert decision.reasoning != ""

    def test_complete_task(self, agent):
        """Test completing a task."""
        task = Task(
            id="task_1",
            description="Write unit tests",
            priority=Priority.MEDIUM
        )

        agent.add_task(task)
        agent.complete_task(task, success=True, tokens_used=5000)

        assert task in agent.completed_tasks
        assert agent.tokens_used == 5000

    def test_get_status(self, agent):
        """Test getting agent status."""
        status = agent.get_status()

        assert "name" in status
        assert "token_budget" in status
        assert "tokens_remaining" in status
        assert "tasks_pending" in status
        assert "success_rate" in status

    def test_executive_summary(self, agent):
        """Test generating executive summary."""
        task = Task(id="t1", description="Test task", priority=Priority.HIGH)
        agent.add_task(task)

        summary = agent.get_executive_summary()

        assert "Executive Summary" in summary
        assert "Resource Status" in summary
        assert "Task Status" in summary


class TestTask:
    """Tests for Task dataclass."""

    def test_task_creation(self):
        """Test creating a task."""
        task = Task(
            id="task_1",
            description="Implement feature",
            priority=Priority.HIGH,
            estimated_effort=7,
            risk_level=RiskLevel.MODERATE
        )

        assert task.id == "task_1"
        assert task.priority == Priority.HIGH
        assert task.estimated_effort == 7
        assert task.risk_level == RiskLevel.MODERATE

    def test_task_defaults(self):
        """Test task default values."""
        task = Task(id="t1", description="Test")

        assert task.priority == Priority.MEDIUM
        assert task.estimated_effort == 1
        assert task.risk_level == RiskLevel.MODERATE
        assert task.dependencies == []


class TestPriority:
    """Tests for Priority enum."""

    def test_priority_ordering(self):
        """Test all priority levels exist."""
        priorities = [
            Priority.CRITICAL,
            Priority.HIGH,
            Priority.MEDIUM,
            Priority.LOW,
            Priority.OPTIONAL
        ]

        assert len(priorities) == 5

    def test_priority_values(self):
        """Test priority string values."""
        assert Priority.CRITICAL.value == "critical"
        assert Priority.HIGH.value == "high"
        assert Priority.MEDIUM.value == "medium"


class TestRiskLevel:
    """Tests for RiskLevel enum."""

    def test_risk_levels(self):
        """Test all risk levels exist."""
        levels = [
            RiskLevel.MINIMAL,
            RiskLevel.LOW,
            RiskLevel.MODERATE,
            RiskLevel.HIGH,
            RiskLevel.CRITICAL
        ]

        assert len(levels) == 5


class TestCEOPrinciples:
    """Tests for CEOPrinciples class."""

    def test_get_principle(self):
        """Test getting a specific principle."""
        principle = CEOPrinciples.get_principle("80_20_rule")

        assert principle != ""
        assert "80%" in principle or "Pareto" in principle

    def test_get_all_principles(self):
        """Test getting all principles."""
        principles = CEOPrinciples.get_all()

        assert isinstance(principles, dict)
        assert len(principles) > 0
        assert "outcome_focus" in principles

    def test_unknown_principle(self):
        """Test getting unknown principle."""
        principle = CEOPrinciples.get_principle("nonexistent")

        assert principle == ""


class TestGetStrategicAgent:
    """Tests for get_strategic_agent factory function."""

    def test_get_agent(self):
        """Test getting default agent."""
        agent = get_strategic_agent(reset=True)

        assert isinstance(agent, StrategicAgent)
        assert agent.name == "RalphCEO"

    def test_get_agent_with_params(self):
        """Test getting agent with custom params."""
        agent = get_strategic_agent(
            name="CustomCEO",
            token_budget=50000,
            reset=True
        )

        assert agent.name == "CustomCEO"
        assert agent.token_budget == 50000

    def test_singleton_behavior(self):
        """Test singleton pattern."""
        agent1 = get_strategic_agent(reset=True)
        agent2 = get_strategic_agent()

        assert agent1 is agent2
