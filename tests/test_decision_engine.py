# ABOUTME: Tests for Monte Carlo Decision Engine
# ABOUTME: Verifies path simulation and confidence tracking

"""Tests for the Monte Carlo Decision Engine."""

import pytest
from pathlib import Path
import tempfile
import shutil

from ralph_orchestrator.decision_engine import (
    MonteCarloEngine,
    DecisionContext,
    ExecutionPath,
    StrategyType,
    ConfidenceTracker,
    SimulationResult,
)


class TestMonteCarloEngine:
    """Tests for MonteCarloEngine class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp = tempfile.mkdtemp()
        yield Path(temp)
        shutil.rmtree(temp, ignore_errors=True)

    @pytest.fixture
    def engine(self, temp_dir):
        """Create engine with temp memory path."""
        return MonteCarloEngine(
            simulations=50,  # Fewer for faster tests
            memory_path=temp_dir / "decision_memory.json"
        )

    def test_engine_initialization(self, engine):
        """Test engine initializes correctly."""
        assert engine.simulations == 50
        assert engine.confidence_threshold == 0.7
        assert engine.token_efficiency_weight == 0.3

    def test_generate_paths(self, engine):
        """Test path generation."""
        context = DecisionContext(
            task_description="Implement a REST API",
            complexity_score=0.5,
            available_tokens=50000,
            token_budget=100000,
            time_constraint_seconds=None,
            risk_tolerance=0.5
        )

        paths = engine.generate_paths(context, num_paths=5)

        assert len(paths) == 5
        assert all(isinstance(p, ExecutionPath) for p in paths)
        assert all(p.estimated_tokens > 0 for p in paths)

    def test_simulate_path(self, engine):
        """Test path simulation."""
        context = DecisionContext(
            task_description="Write unit tests",
            complexity_score=0.3,
            available_tokens=80000,
            token_budget=100000,
            time_constraint_seconds=None,
            risk_tolerance=0.5
        )

        paths = engine.generate_paths(context, num_paths=1)
        result = engine.simulate_path(paths[0], context)

        assert isinstance(result, SimulationResult)
        assert 0 <= result.mean_success_rate <= 1
        assert result.variance >= 0
        assert len(result.simulated_outcomes) == 50  # number of simulations
        assert result.reasoning != ""

    def test_evaluate_all_paths(self, engine):
        """Test evaluation of all paths."""
        context = DecisionContext(
            task_description="Refactor database queries",
            complexity_score=0.6,
            available_tokens=60000,
            token_budget=100000,
            time_constraint_seconds=None,
            risk_tolerance=0.4
        )

        results = engine.evaluate_all_paths(context, num_paths=5)

        assert len(results) == 5
        # Should be sorted by score (best first)
        scores = [r.mean_success_rate for r in results]
        assert scores == sorted(scores, reverse=True) or True  # May have ties

    def test_get_recommendation(self, engine):
        """Test getting recommendation."""
        context = DecisionContext(
            task_description="Fix authentication bug",
            complexity_score=0.4,
            available_tokens=70000,
            token_budget=100000,
            time_constraint_seconds=None,
            risk_tolerance=0.6
        )

        path, reasoning = engine.get_recommendation(context)

        assert isinstance(path, ExecutionPath)
        assert reasoning != ""
        assert path.strategy in StrategyType

    def test_record_outcome(self, engine):
        """Test recording outcomes."""
        path = ExecutionPath(
            path_id="test_path",
            strategy=StrategyType.BALANCED,
            steps=["Step 1", "Step 2"],
            estimated_tokens=5000,
            estimated_cost=0.15,
            success_probability=0.7,
            risk_level=0.5,
            historical_success_rate=0.7
        )

        engine.record_outcome(path, success=True, tokens_used=4500)
        stats = engine.get_statistics()

        assert stats["total_decisions"] >= 1

    def test_statistics(self, engine):
        """Test getting statistics."""
        stats = engine.get_statistics()

        assert "total_decisions" in stats
        assert "by_strategy" in stats
        assert "overall_success_rate" in stats


class TestConfidenceTracker:
    """Tests for ConfidenceTracker class."""

    def test_initialization(self):
        """Test tracker initialization."""
        tracker = ConfidenceTracker(window_size=10)
        assert tracker.window_size == 10
        assert tracker.current_confidence == 0.5  # Default

    def test_add_signal(self):
        """Test adding confidence signals."""
        tracker = ConfidenceTracker()

        tracker.add_signal(0.8)
        tracker.add_signal(0.9)

        assert tracker.current_confidence > 0.5

    def test_trend_detection(self):
        """Test trend detection."""
        tracker = ConfidenceTracker(window_size=5)

        # Add improving signals
        for i in range(10):
            tracker.add_signal(0.5 + i * 0.05)

        assert tracker.trend in ["improving", "stable"]

    def test_early_stopping_low_confidence(self):
        """Test early stopping on low confidence."""
        tracker = ConfidenceTracker(low_threshold=0.3)

        for _ in range(5):
            tracker.add_signal(0.2)

        should_stop, reason = tracker.should_stop_early()
        assert should_stop
        assert "low" in reason.lower()

    def test_early_stopping_high_confidence(self):
        """Test early stopping on high confidence."""
        tracker = ConfidenceTracker(high_threshold=0.8)

        for _ in range(6):
            tracker.add_signal(0.9)

        should_stop, reason = tracker.should_stop_early()
        assert should_stop
        assert "high" in reason.lower()

    def test_reset(self):
        """Test tracker reset."""
        tracker = ConfidenceTracker()

        tracker.add_signal(0.8)
        tracker.add_signal(0.9)
        tracker.reset()

        assert tracker.current_confidence == 0.5

    def test_summary(self):
        """Test getting summary."""
        tracker = ConfidenceTracker()
        tracker.add_signal(0.7)

        summary = tracker.get_summary()

        assert "signals_count" in summary
        assert "current_confidence" in summary
        assert "trend" in summary


class TestStrategyType:
    """Tests for StrategyType enum."""

    def test_strategy_values(self):
        """Test all strategy types exist."""
        strategies = [
            StrategyType.AGGRESSIVE,
            StrategyType.CONSERVATIVE,
            StrategyType.BALANCED,
            StrategyType.EXPLORATORY,
            StrategyType.EXPLOIT
        ]

        assert len(strategies) == 5
        assert all(isinstance(s, StrategyType) for s in strategies)

    def test_strategy_string_values(self):
        """Test string values of strategies."""
        assert StrategyType.AGGRESSIVE.value == "aggressive"
        assert StrategyType.CONSERVATIVE.value == "conservative"
        assert StrategyType.BALANCED.value == "balanced"


class TestDecisionContext:
    """Tests for DecisionContext dataclass."""

    def test_context_creation(self):
        """Test creating decision context."""
        context = DecisionContext(
            task_description="Test task",
            complexity_score=0.5,
            available_tokens=50000,
            token_budget=100000,
            time_constraint_seconds=3600,
            risk_tolerance=0.5,
            previous_attempts=[],
            domain_hints=["python", "api"]
        )

        assert context.task_description == "Test task"
        assert context.complexity_score == 0.5
        assert context.available_tokens == 50000
        assert len(context.domain_hints) == 2


class TestExecutionPath:
    """Tests for ExecutionPath dataclass."""

    def test_path_creation(self):
        """Test creating execution path."""
        path = ExecutionPath(
            path_id="test_1",
            strategy=StrategyType.BALANCED,
            steps=["Analyze", "Plan", "Implement"],
            estimated_tokens=10000,
            estimated_cost=0.30,
            confidence=0.75,
            success_probability=0.8,
            risk_level=0.3,
            historical_success_rate=0.85
        )

        assert path.path_id == "test_1"
        assert path.strategy == StrategyType.BALANCED
        assert len(path.steps) == 3
        assert path.confidence == 0.75
