# ABOUTME: Monte Carlo Decision Engine for optimal path finding
# ABOUTME: Implements DeepConf-inspired confidence tracking and path optimization

"""Monte Carlo Decision Engine for Ralph Orchestrator.

This module implements a lightweight Monte Carlo simulation engine for
evaluating different execution paths and strategies. Inspired by DeepConf's
confidence-based filtering approach for token efficiency.
"""

import asyncio
import logging
import random
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple
from enum import Enum
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class StrategyType(str, Enum):
    """Types of strategies the engine can evaluate."""
    AGGRESSIVE = "aggressive"      # Fast execution, higher risk
    CONSERVATIVE = "conservative"  # Slow, careful execution
    BALANCED = "balanced"          # Middle ground
    EXPLORATORY = "exploratory"    # Try new approaches
    EXPLOIT = "exploit"            # Use known good approaches


@dataclass
class ExecutionPath:
    """Represents a possible execution path with its characteristics."""
    path_id: str
    strategy: StrategyType
    steps: List[str]
    estimated_tokens: int
    estimated_cost: float
    confidence: float = 0.0
    success_probability: float = 0.5
    risk_level: float = 0.5
    historical_success_rate: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationResult:
    """Result of a Monte Carlo simulation run."""
    path: ExecutionPath
    simulated_outcomes: List[bool]
    mean_success_rate: float
    variance: float
    confidence_interval: Tuple[float, float]
    token_efficiency_score: float
    recommended: bool
    reasoning: str


@dataclass
class DecisionContext:
    """Context for making decisions."""
    task_description: str
    complexity_score: float  # 0-1
    available_tokens: int
    token_budget: int
    time_constraint_seconds: Optional[int]
    risk_tolerance: float  # 0-1, higher = more risk tolerant
    previous_attempts: List[Dict[str, Any]] = field(default_factory=list)
    domain_hints: List[str] = field(default_factory=list)


class MonteCarloEngine:
    """Monte Carlo simulation engine for decision optimization.

    Uses statistical simulations to evaluate different execution paths
    and recommend optimal strategies based on historical data and
    confidence metrics.
    """

    # Default simulation parameters
    DEFAULT_SIMULATIONS = 100
    DEFAULT_CONFIDENCE_THRESHOLD = 0.7
    DEFAULT_TOKEN_EFFICIENCY_WEIGHT = 0.3

    def __init__(
        self,
        simulations: int = DEFAULT_SIMULATIONS,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        token_efficiency_weight: float = DEFAULT_TOKEN_EFFICIENCY_WEIGHT,
        memory_path: Optional[Path] = None
    ):
        """Initialize the Monte Carlo engine.

        Args:
            simulations: Number of simulations per path evaluation
            confidence_threshold: Minimum confidence to recommend a path
            token_efficiency_weight: Weight for token efficiency in scoring
            memory_path: Path to persistent memory file
        """
        self.simulations = simulations
        self.confidence_threshold = confidence_threshold
        self.token_efficiency_weight = token_efficiency_weight
        self.memory_path = memory_path or Path(".agent/decision_memory.json")

        # Internal state
        self._historical_data: Dict[str, List[Dict]] = {}
        self._pattern_cache: Dict[str, float] = {}
        self._load_memory()

    def _load_memory(self) -> None:
        """Load historical decision data from persistent storage."""
        if self.memory_path.exists():
            try:
                data = json.loads(self.memory_path.read_text())
                self._historical_data = data.get("historical_data", {})
                self._pattern_cache = data.get("pattern_cache", {})
                logger.debug(f"Loaded {len(self._historical_data)} historical patterns")
            except Exception as e:
                logger.warning(f"Failed to load decision memory: {e}")

    def _save_memory(self) -> None:
        """Save historical decision data to persistent storage."""
        try:
            self.memory_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "historical_data": self._historical_data,
                "pattern_cache": self._pattern_cache,
                "last_updated": datetime.now().isoformat()
            }
            self.memory_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning(f"Failed to save decision memory: {e}")

    def generate_paths(
        self,
        context: DecisionContext,
        num_paths: int = 5
    ) -> List[ExecutionPath]:
        """Generate possible execution paths for a given context.

        Args:
            context: Decision context with task info and constraints
            num_paths: Number of paths to generate

        Returns:
            List of possible execution paths
        """
        paths = []

        # Generate paths for each strategy type
        strategies = list(StrategyType)

        for i in range(num_paths):
            strategy = strategies[i % len(strategies)]

            # Calculate base parameters based on strategy
            if strategy == StrategyType.AGGRESSIVE:
                token_estimate = int(context.token_budget * 0.3)
                risk = 0.7
                success_base = 0.6
            elif strategy == StrategyType.CONSERVATIVE:
                token_estimate = int(context.token_budget * 0.8)
                risk = 0.2
                success_base = 0.8
            elif strategy == StrategyType.BALANCED:
                token_estimate = int(context.token_budget * 0.5)
                risk = 0.5
                success_base = 0.7
            elif strategy == StrategyType.EXPLORATORY:
                token_estimate = int(context.token_budget * 0.6)
                risk = 0.6
                success_base = 0.5
            else:  # EXPLOIT
                token_estimate = int(context.token_budget * 0.4)
                risk = 0.3
                success_base = 0.75

            # Adjust based on complexity
            complexity_factor = 1 + (context.complexity_score * 0.5)
            token_estimate = int(token_estimate * complexity_factor)

            # Look up historical success rate for similar tasks
            historical_rate = self._get_historical_success_rate(
                context.task_description,
                strategy
            )

            path = ExecutionPath(
                path_id=f"path_{i}_{strategy.value}",
                strategy=strategy,
                steps=self._generate_steps(context, strategy),
                estimated_tokens=token_estimate,
                estimated_cost=token_estimate * 0.00003,  # Rough estimate
                confidence=0.0,  # Will be set by simulation
                success_probability=success_base,
                risk_level=risk,
                historical_success_rate=historical_rate,
                metadata={
                    "complexity_factor": complexity_factor,
                    "generated_at": datetime.now().isoformat()
                }
            )
            paths.append(path)

        return paths

    def _generate_steps(
        self,
        context: DecisionContext,
        strategy: StrategyType
    ) -> List[str]:
        """Generate step sequence for a strategy.

        Args:
            context: Decision context
            strategy: Strategy type

        Returns:
            List of step descriptions
        """
        base_steps = [
            "Analyze task requirements",
            "Identify key components",
            "Plan implementation approach"
        ]

        if strategy == StrategyType.AGGRESSIVE:
            return base_steps + [
                "Implement core functionality directly",
                "Run minimal validation"
            ]
        elif strategy == StrategyType.CONSERVATIVE:
            return base_steps + [
                "Review existing codebase",
                "Write comprehensive tests first",
                "Implement with TDD approach",
                "Run full test suite",
                "Document changes"
            ]
        elif strategy == StrategyType.BALANCED:
            return base_steps + [
                "Write key tests",
                "Implement functionality",
                "Validate with tests"
            ]
        elif strategy == StrategyType.EXPLORATORY:
            return base_steps + [
                "Research alternative approaches",
                "Prototype solution",
                "Evaluate and refine",
                "Implement final version"
            ]
        else:  # EXPLOIT
            return base_steps + [
                "Apply known patterns",
                "Implement using proven approach",
                "Verify results"
            ]

    def _get_historical_success_rate(
        self,
        task_description: str,
        strategy: StrategyType
    ) -> float:
        """Get historical success rate for similar tasks.

        Args:
            task_description: Description of the task
            strategy: Strategy type

        Returns:
            Historical success rate (0-1)
        """
        # Create a simple pattern key from the task
        words = task_description.lower().split()
        key_words = [w for w in words if len(w) > 4][:5]
        pattern_key = f"{strategy.value}:{'_'.join(key_words)}"

        # Check cache
        if pattern_key in self._pattern_cache:
            return self._pattern_cache[pattern_key]

        # Check historical data
        if strategy.value in self._historical_data:
            history = self._historical_data[strategy.value]
            if history:
                successes = sum(1 for h in history if h.get("success", False))
                return successes / len(history)

        # Default success rate based on strategy
        defaults = {
            StrategyType.AGGRESSIVE: 0.6,
            StrategyType.CONSERVATIVE: 0.8,
            StrategyType.BALANCED: 0.7,
            StrategyType.EXPLORATORY: 0.5,
            StrategyType.EXPLOIT: 0.75
        }
        return defaults.get(strategy, 0.5)

    def simulate_path(
        self,
        path: ExecutionPath,
        context: DecisionContext
    ) -> SimulationResult:
        """Run Monte Carlo simulation for a single path.

        Args:
            path: Execution path to simulate
            context: Decision context

        Returns:
            Simulation result with statistics
        """
        outcomes = []

        # Run simulations
        for _ in range(self.simulations):
            # Calculate success probability for this run
            base_prob = path.success_probability

            # Adjust for historical data
            historical_adjustment = (path.historical_success_rate - 0.5) * 0.3

            # Adjust for risk tolerance
            risk_adjustment = (context.risk_tolerance - 0.5) * path.risk_level * 0.2

            # Adjust for complexity
            complexity_penalty = context.complexity_score * 0.1

            # Random variation
            variation = random.gauss(0, 0.1)

            # Final probability
            final_prob = base_prob + historical_adjustment + risk_adjustment - complexity_penalty + variation
            final_prob = max(0.1, min(0.95, final_prob))  # Clamp to reasonable range

            # Simulate outcome
            outcome = random.random() < final_prob
            outcomes.append(outcome)

        # Calculate statistics
        success_count = sum(outcomes)
        mean_success = success_count / self.simulations

        # Calculate variance
        variance = sum((1 if o else 0 - mean_success) ** 2 for o in outcomes) / self.simulations
        std_dev = math.sqrt(variance)

        # 95% confidence interval
        z = 1.96  # 95% confidence
        margin = z * std_dev / math.sqrt(self.simulations)
        confidence_interval = (
            max(0, mean_success - margin),
            min(1, mean_success + margin)
        )

        # Calculate token efficiency score
        token_efficiency = 1 - (path.estimated_tokens / context.token_budget)
        token_efficiency = max(0, token_efficiency)

        # Update path confidence
        path.confidence = mean_success

        # Determine recommendation
        composite_score = (
            mean_success * (1 - self.token_efficiency_weight) +
            token_efficiency * self.token_efficiency_weight
        )
        recommended = composite_score >= self.confidence_threshold

        # Generate reasoning
        reasoning = self._generate_reasoning(
            path,
            mean_success,
            token_efficiency,
            context
        )

        return SimulationResult(
            path=path,
            simulated_outcomes=outcomes,
            mean_success_rate=mean_success,
            variance=variance,
            confidence_interval=confidence_interval,
            token_efficiency_score=token_efficiency,
            recommended=recommended,
            reasoning=reasoning
        )

    def _generate_reasoning(
        self,
        path: ExecutionPath,
        success_rate: float,
        token_efficiency: float,
        context: DecisionContext
    ) -> str:
        """Generate human-readable reasoning for the recommendation.

        Args:
            path: Evaluated execution path
            success_rate: Simulated success rate
            token_efficiency: Token efficiency score
            context: Decision context

        Returns:
            Reasoning string
        """
        reasons = []

        if success_rate >= 0.8:
            reasons.append(f"High success probability ({success_rate:.1%})")
        elif success_rate >= 0.6:
            reasons.append(f"Moderate success probability ({success_rate:.1%})")
        else:
            reasons.append(f"Lower success probability ({success_rate:.1%}) - consider alternatives")

        if token_efficiency >= 0.5:
            reasons.append(f"Good token efficiency ({token_efficiency:.1%} budget remaining)")
        elif token_efficiency >= 0.2:
            reasons.append(f"Moderate token efficiency ({token_efficiency:.1%} budget remaining)")
        else:
            reasons.append(f"High token usage expected")

        if path.historical_success_rate >= 0.7:
            reasons.append(f"Strong historical performance ({path.historical_success_rate:.1%})")

        strategy_notes = {
            StrategyType.AGGRESSIVE: "Fast but risky approach",
            StrategyType.CONSERVATIVE: "Thorough, safer approach",
            StrategyType.BALANCED: "Balanced risk/reward",
            StrategyType.EXPLORATORY: "May discover better approaches",
            StrategyType.EXPLOIT: "Uses proven patterns"
        }
        reasons.append(strategy_notes.get(path.strategy, ""))

        return ". ".join(reasons)

    def evaluate_all_paths(
        self,
        context: DecisionContext,
        num_paths: int = 5
    ) -> List[SimulationResult]:
        """Evaluate all possible paths and return sorted results.

        Args:
            context: Decision context
            num_paths: Number of paths to generate and evaluate

        Returns:
            List of simulation results, sorted by recommendation score
        """
        paths = self.generate_paths(context, num_paths)
        results = []

        for path in paths:
            result = self.simulate_path(path, context)
            results.append(result)

        # Sort by composite score (success rate + token efficiency)
        results.sort(
            key=lambda r: (
                r.mean_success_rate * (1 - self.token_efficiency_weight) +
                r.token_efficiency_score * self.token_efficiency_weight
            ),
            reverse=True
        )

        return results

    def get_recommendation(
        self,
        context: DecisionContext,
        num_paths: int = 5
    ) -> Tuple[ExecutionPath, str]:
        """Get the recommended execution path for a task.

        Args:
            context: Decision context
            num_paths: Number of paths to evaluate

        Returns:
            Tuple of (recommended path, reasoning)
        """
        results = self.evaluate_all_paths(context, num_paths)

        if not results:
            # Fallback to balanced strategy
            default_path = ExecutionPath(
                path_id="default_balanced",
                strategy=StrategyType.BALANCED,
                steps=["Analyze", "Plan", "Implement", "Verify"],
                estimated_tokens=context.token_budget // 2,
                estimated_cost=0.0,
                confidence=0.5,
                success_probability=0.7,
                risk_level=0.5,
                historical_success_rate=0.7
            )
            return default_path, "Using default balanced strategy"

        best = results[0]
        return best.path, best.reasoning

    def record_outcome(
        self,
        path: ExecutionPath,
        success: bool,
        tokens_used: int,
        notes: str = ""
    ) -> None:
        """Record the actual outcome for learning.

        Args:
            path: The path that was executed
            success: Whether it succeeded
            tokens_used: Actual tokens used
            notes: Additional notes
        """
        strategy_key = path.strategy.value

        if strategy_key not in self._historical_data:
            self._historical_data[strategy_key] = []

        self._historical_data[strategy_key].append({
            "path_id": path.path_id,
            "success": success,
            "predicted_success": path.success_probability,
            "tokens_estimated": path.estimated_tokens,
            "tokens_used": tokens_used,
            "timestamp": datetime.now().isoformat(),
            "notes": notes
        })

        # Limit history size per strategy
        if len(self._historical_data[strategy_key]) > 1000:
            self._historical_data[strategy_key] = self._historical_data[strategy_key][-500:]

        # Update pattern cache
        if path.metadata.get("pattern_key"):
            pattern_key = path.metadata["pattern_key"]
            history = self._historical_data[strategy_key]
            recent = history[-100:] if len(history) > 100 else history
            successes = sum(1 for h in recent if h.get("success", False))
            self._pattern_cache[pattern_key] = successes / len(recent) if recent else 0.5

        self._save_memory()
        logger.info(f"Recorded outcome: {path.strategy.value} - {'success' if success else 'failure'}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics.

        Returns:
            Statistics dictionary
        """
        stats = {
            "total_decisions": 0,
            "by_strategy": {},
            "overall_success_rate": 0.0,
            "avg_token_accuracy": 0.0
        }

        all_outcomes = []
        token_accuracies = []

        for strategy, history in self._historical_data.items():
            successes = sum(1 for h in history if h.get("success", False))
            strategy_rate = successes / len(history) if history else 0

            stats["by_strategy"][strategy] = {
                "total": len(history),
                "successes": successes,
                "success_rate": strategy_rate
            }
            stats["total_decisions"] += len(history)
            all_outcomes.extend([h.get("success", False) for h in history])

            # Calculate token accuracy
            for h in history:
                est = h.get("tokens_estimated", 0)
                actual = h.get("tokens_used", 0)
                if est > 0 and actual > 0:
                    accuracy = 1 - abs(est - actual) / max(est, actual)
                    token_accuracies.append(accuracy)

        if all_outcomes:
            stats["overall_success_rate"] = sum(all_outcomes) / len(all_outcomes)

        if token_accuracies:
            stats["avg_token_accuracy"] = sum(token_accuracies) / len(token_accuracies)

        return stats


class ConfidenceTracker:
    """DeepConf-inspired confidence tracking for token efficiency.

    Tracks confidence metrics during execution to enable early stopping
    of unpromising paths, similar to DeepConf's approach.
    """

    def __init__(
        self,
        window_size: int = 10,
        low_threshold: float = 0.3,
        high_threshold: float = 0.8
    ):
        """Initialize confidence tracker.

        Args:
            window_size: Number of recent items to track for rolling confidence
            low_threshold: Below this, consider path failing
            high_threshold: Above this, consider path succeeding
        """
        self.window_size = window_size
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

        self._signals: List[float] = []
        self._cumulative_confidence = 0.0

    def add_signal(self, confidence: float) -> None:
        """Add a confidence signal.

        Args:
            confidence: Confidence value (0-1)
        """
        self._signals.append(confidence)
        self._cumulative_confidence = sum(self._signals) / len(self._signals)

    @property
    def current_confidence(self) -> float:
        """Get current rolling confidence."""
        if not self._signals:
            return 0.5

        recent = self._signals[-self.window_size:]
        return sum(recent) / len(recent)

    @property
    def cumulative_confidence(self) -> float:
        """Get overall cumulative confidence."""
        return self._cumulative_confidence

    @property
    def trend(self) -> str:
        """Get confidence trend."""
        if len(self._signals) < 2:
            return "stable"

        recent = self._signals[-self.window_size:]
        if len(recent) < 2:
            return "stable"

        first_half = sum(recent[:len(recent)//2]) / (len(recent)//2)
        second_half = sum(recent[len(recent)//2:]) / (len(recent) - len(recent)//2)

        diff = second_half - first_half
        if diff > 0.1:
            return "improving"
        elif diff < -0.1:
            return "declining"
        return "stable"

    def should_stop_early(self) -> Tuple[bool, str]:
        """Check if we should stop early based on confidence.

        Returns:
            Tuple of (should_stop, reason)
        """
        if len(self._signals) < 3:
            return False, "Insufficient data"

        current = self.current_confidence

        if current < self.low_threshold:
            return True, f"Confidence too low ({current:.2f} < {self.low_threshold})"

        if current > self.high_threshold and len(self._signals) > 5:
            return True, f"High confidence achieved ({current:.2f} > {self.high_threshold})"

        # Check for stagnation
        if len(self._signals) > 10:
            variance = sum((s - current) ** 2 for s in self._signals[-10:]) / 10
            if variance < 0.01 and current < 0.5:
                return True, f"Confidence stagnated at low level ({current:.2f})"

        return False, "Continue execution"

    def reset(self) -> None:
        """Reset tracker state."""
        self._signals = []
        self._cumulative_confidence = 0.0

    def get_summary(self) -> Dict[str, Any]:
        """Get tracker summary."""
        return {
            "signals_count": len(self._signals),
            "current_confidence": self.current_confidence,
            "cumulative_confidence": self.cumulative_confidence,
            "trend": self.trend,
            "window_size": self.window_size,
            "thresholds": {
                "low": self.low_threshold,
                "high": self.high_threshold
            }
        }
