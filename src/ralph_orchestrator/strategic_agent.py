# ABOUTME: CEO/Strategic Agent for high-level decision making and orchestration
# ABOUTME: Acts as the unified entry point with successful CEO mindset

"""Strategic Agent for Ralph Orchestrator.

This module implements a high-level strategic agent that acts as a
"successful CEO" advisor, making decisions about task prioritization,
resource allocation, risk management, and execution strategy.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from datetime import datetime
import json
from pathlib import Path

from .decision_engine import (
    MonteCarloEngine,
    DecisionContext,
    ExecutionPath,
    StrategyType,
    ConfidenceTracker
)

logger = logging.getLogger(__name__)


class Priority(str, Enum):
    """Task priority levels."""
    CRITICAL = "critical"      # Must do immediately
    HIGH = "high"              # Important, do soon
    MEDIUM = "medium"          # Standard priority
    LOW = "low"                # Can wait
    OPTIONAL = "optional"      # Nice to have


class RiskLevel(str, Enum):
    """Risk assessment levels."""
    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Task:
    """Represents a task to be executed."""
    id: str
    description: str
    priority: Priority = Priority.MEDIUM
    estimated_effort: int = 1  # 1-10 scale
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.MODERATE
    deadline: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategicDecision:
    """A strategic decision made by the agent."""
    decision_id: str
    task: Task
    recommended_path: ExecutionPath
    confidence: float
    reasoning: str
    alternatives: List[ExecutionPath]
    risk_analysis: Dict[str, Any]
    resource_allocation: Dict[str, int]
    expected_outcome: str
    timestamp: datetime = field(default_factory=datetime.now)


class CEOPrinciples:
    """Core principles that guide CEO-level decision making.

    Based on successful CEO mindsets:
    1. Outcome-focused thinking
    2. Risk-aware but not risk-averse
    3. Resource optimization
    4. Long-term value creation
    5. Decisive action with measured confidence
    """

    PRINCIPLES = {
        "outcome_focus": (
            "Focus on the desired outcome, not just the process. "
            "What does success look like? Work backwards from there."
        ),
        "80_20_rule": (
            "Apply Pareto principle: 20% of efforts produce 80% of results. "
            "Identify and prioritize high-impact activities."
        ),
        "calculated_risk": (
            "Take calculated risks with high potential returns. "
            "Avoid unnecessary risks, but don't let fear of failure paralyze action."
        ),
        "resource_efficiency": (
            "Maximize ROI on resources (time, tokens, compute). "
            "Don't spend $100 on a $10 problem."
        ),
        "iterate_fast": (
            "Prefer fast iterations over perfect first attempts. "
            "Learn from failures quickly and adjust."
        ),
        "compound_thinking": (
            "Consider how decisions compound over time. "
            "Small improvements now lead to big gains later."
        ),
        "decisive_action": (
            "Make decisions with incomplete information when necessary. "
            "A good decision now is better than a perfect decision later."
        )
    }

    @classmethod
    def get_principle(cls, key: str) -> str:
        """Get a specific principle."""
        return cls.PRINCIPLES.get(key, "")

    @classmethod
    def get_all(cls) -> Dict[str, str]:
        """Get all principles."""
        return cls.PRINCIPLES.copy()


class StrategicAgent:
    """High-level strategic agent with CEO mindset.

    This agent provides:
    - Task prioritization and scheduling
    - Resource allocation decisions
    - Risk assessment and mitigation
    - Strategy selection using Monte Carlo simulation
    - Performance tracking and optimization
    """

    def __init__(
        self,
        name: str = "RalphCEO",
        token_budget: int = 100000,
        risk_tolerance: float = 0.5,
        memory_path: Optional[Path] = None
    ):
        """Initialize the Strategic Agent.

        Args:
            name: Agent's identifier
            token_budget: Total token budget for operations
            risk_tolerance: Risk tolerance level (0-1)
            memory_path: Path for persistent memory
        """
        self.name = name
        self.token_budget = token_budget
        self.tokens_used = 0
        self.risk_tolerance = risk_tolerance
        self.memory_path = memory_path or Path(".agent/strategic_memory.json")

        # Initialize components
        self.decision_engine = MonteCarloEngine(
            memory_path=self.memory_path.parent / "decision_memory.json"
        )
        self.confidence_tracker = ConfidenceTracker()

        # State
        self.task_queue: List[Task] = []
        self.completed_tasks: List[Task] = []
        self.decision_history: List[StrategicDecision] = []
        self.performance_metrics: Dict[str, Any] = {}

        # Load memory
        self._load_memory()

        logger.info(f"Strategic Agent '{name}' initialized")

    def _load_memory(self) -> None:
        """Load persistent memory."""
        if self.memory_path.exists():
            try:
                data = json.loads(self.memory_path.read_text())
                self.performance_metrics = data.get("performance_metrics", {})
                logger.debug(f"Loaded strategic memory")
            except Exception as e:
                logger.warning(f"Failed to load strategic memory: {e}")

    def _save_memory(self) -> None:
        """Save persistent memory."""
        try:
            self.memory_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "performance_metrics": self.performance_metrics,
                "last_updated": datetime.now().isoformat(),
                "agent_name": self.name,
                "total_decisions": len(self.decision_history),
                "completed_tasks": len(self.completed_tasks)
            }
            self.memory_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning(f"Failed to save strategic memory: {e}")

    def add_task(self, task: Task) -> None:
        """Add a task to the queue.

        Args:
            task: Task to add
        """
        self.task_queue.append(task)
        self._prioritize_queue()
        logger.info(f"Added task: {task.id} (Priority: {task.priority.value})")

    def add_tasks_from_prompt(self, prompt: str) -> List[Task]:
        """Extract and add tasks from a prompt.

        Args:
            prompt: Text containing task descriptions

        Returns:
            List of created tasks
        """
        tasks = []
        lines = prompt.strip().split('\n')

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # Parse common task formats
            # - [ ] Task description
            # - Task description
            # 1. Task description
            # TODO: Task description

            task_desc = None
            if line.startswith('- [ ]'):
                task_desc = line[5:].strip()
            elif line.startswith('- [x]'):
                continue  # Skip completed
            elif line.startswith('- '):
                task_desc = line[2:].strip()
            elif line[0].isdigit() and '. ' in line:
                task_desc = line.split('. ', 1)[1].strip()
            elif line.lower().startswith('todo:'):
                task_desc = line[5:].strip()

            if task_desc:
                # Estimate priority and effort based on keywords
                priority = self._estimate_priority(task_desc)
                effort = self._estimate_effort(task_desc)
                risk = self._estimate_risk(task_desc)

                task = Task(
                    id=f"task_{i}_{datetime.now().strftime('%H%M%S')}",
                    description=task_desc,
                    priority=priority,
                    estimated_effort=effort,
                    risk_level=risk
                )
                tasks.append(task)
                self.add_task(task)

        return tasks

    def _estimate_priority(self, description: str) -> Priority:
        """Estimate task priority from description.

        Args:
            description: Task description

        Returns:
            Estimated priority
        """
        desc_lower = description.lower()

        critical_keywords = ['critical', 'urgent', 'asap', 'immediately', 'blocker', 'security']
        high_keywords = ['important', 'high priority', 'should', 'required', 'must']
        low_keywords = ['optional', 'nice to have', 'consider', 'maybe', 'could']

        for keyword in critical_keywords:
            if keyword in desc_lower:
                return Priority.CRITICAL

        for keyword in high_keywords:
            if keyword in desc_lower:
                return Priority.HIGH

        for keyword in low_keywords:
            if keyword in desc_lower:
                return Priority.LOW

        return Priority.MEDIUM

    def _estimate_effort(self, description: str) -> int:
        """Estimate task effort from description.

        Args:
            description: Task description

        Returns:
            Estimated effort (1-10)
        """
        desc_lower = description.lower()

        # Count complexity indicators
        complexity = 5  # Default

        high_effort = ['refactor', 'redesign', 'implement', 'create', 'build', 'migrate']
        medium_effort = ['update', 'modify', 'add', 'change', 'integrate']
        low_effort = ['fix', 'adjust', 'tweak', 'rename', 'remove']

        for keyword in high_effort:
            if keyword in desc_lower:
                complexity += 2
                break

        for keyword in medium_effort:
            if keyword in desc_lower:
                complexity += 1
                break

        for keyword in low_effort:
            if keyword in desc_lower:
                complexity -= 1
                break

        # Clamp to 1-10
        return max(1, min(10, complexity))

    def _estimate_risk(self, description: str) -> RiskLevel:
        """Estimate task risk from description.

        Args:
            description: Task description

        Returns:
            Estimated risk level
        """
        desc_lower = description.lower()

        high_risk = ['database', 'production', 'security', 'auth', 'payment', 'delete', 'migration']
        moderate_risk = ['api', 'integration', 'refactor', 'upgrade', 'dependency']
        low_risk = ['docs', 'test', 'comment', 'format', 'style', 'readme']

        for keyword in high_risk:
            if keyword in desc_lower:
                return RiskLevel.HIGH

        for keyword in moderate_risk:
            if keyword in desc_lower:
                return RiskLevel.MODERATE

        for keyword in low_risk:
            if keyword in desc_lower:
                return RiskLevel.LOW

        return RiskLevel.MODERATE

    def _prioritize_queue(self) -> None:
        """Sort task queue by priority and dependencies."""
        # Priority order: CRITICAL > HIGH > MEDIUM > LOW > OPTIONAL
        priority_order = {
            Priority.CRITICAL: 0,
            Priority.HIGH: 1,
            Priority.MEDIUM: 2,
            Priority.LOW: 3,
            Priority.OPTIONAL: 4
        }

        self.task_queue.sort(
            key=lambda t: (
                priority_order[t.priority],
                -t.estimated_effort,  # Higher effort tasks first within priority
                t.created_at
            )
        )

    def get_next_task(self) -> Optional[Task]:
        """Get the next task to execute.

        Returns:
            Next task or None if queue is empty
        """
        if not self.task_queue:
            return None

        # Check dependencies
        for task in self.task_queue:
            if not task.dependencies:
                return task

            # Check if all dependencies are completed
            completed_ids = {t.id for t in self.completed_tasks}
            if all(dep in completed_ids for dep in task.dependencies):
                return task

        # If all tasks have unmet dependencies, return first one anyway
        return self.task_queue[0] if self.task_queue else None

    def make_decision(self, task: Task) -> StrategicDecision:
        """Make a strategic decision about how to execute a task.

        Uses Monte Carlo simulation to evaluate paths and applies
        CEO principles to select optimal strategy.

        Args:
            task: Task to make decision for

        Returns:
            Strategic decision with recommended approach
        """
        # Create decision context
        context = DecisionContext(
            task_description=task.description,
            complexity_score=task.estimated_effort / 10,
            available_tokens=self.token_budget - self.tokens_used,
            token_budget=self.token_budget,
            time_constraint_seconds=None,
            risk_tolerance=self._adjust_risk_tolerance(task),
            previous_attempts=[],
            domain_hints=task.tags
        )

        # Get Monte Carlo simulation results
        results = self.decision_engine.evaluate_all_paths(context)

        if not results:
            # Fallback to balanced approach
            path = ExecutionPath(
                path_id="default",
                strategy=StrategyType.BALANCED,
                steps=["Analyze", "Plan", "Implement", "Verify"],
                estimated_tokens=context.available_tokens // 2,
                estimated_cost=0.0,
                confidence=0.5,
                success_probability=0.7,
                risk_level=0.5,
                historical_success_rate=0.7
            )
            results = [type('Result', (), {
                'path': path,
                'mean_success_rate': 0.7,
                'token_efficiency_score': 0.5,
                'reasoning': 'Default balanced strategy'
            })()]

        best_result = results[0]

        # Risk analysis
        risk_analysis = self._analyze_risk(task, best_result.path)

        # Resource allocation
        resource_allocation = self._allocate_resources(task, best_result.path)

        # Create decision
        decision = StrategicDecision(
            decision_id=f"dec_{task.id}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            task=task,
            recommended_path=best_result.path,
            confidence=best_result.mean_success_rate,
            reasoning=self._generate_ceo_reasoning(task, best_result),
            alternatives=[r.path for r in results[1:4]],
            risk_analysis=risk_analysis,
            resource_allocation=resource_allocation,
            expected_outcome=self._predict_outcome(task, best_result)
        )

        self.decision_history.append(decision)
        logger.info(f"Decision made for {task.id}: {best_result.path.strategy.value} "
                   f"(confidence: {decision.confidence:.2%})")

        return decision

    def _adjust_risk_tolerance(self, task: Task) -> float:
        """Adjust risk tolerance based on task characteristics.

        Args:
            task: Current task

        Returns:
            Adjusted risk tolerance (0-1)
        """
        base = self.risk_tolerance

        # Reduce risk tolerance for critical tasks
        if task.priority == Priority.CRITICAL:
            base *= 0.7
        elif task.priority == Priority.HIGH:
            base *= 0.85

        # Reduce for high-risk tasks
        risk_adjustments = {
            RiskLevel.MINIMAL: 1.1,
            RiskLevel.LOW: 1.0,
            RiskLevel.MODERATE: 0.9,
            RiskLevel.HIGH: 0.7,
            RiskLevel.CRITICAL: 0.5
        }
        base *= risk_adjustments.get(task.risk_level, 1.0)

        # Consider remaining budget
        budget_ratio = (self.token_budget - self.tokens_used) / self.token_budget
        if budget_ratio < 0.2:
            base *= 0.6  # Be more conservative when low on budget

        return max(0.1, min(0.9, base))

    def _analyze_risk(
        self,
        task: Task,
        path: ExecutionPath
    ) -> Dict[str, Any]:
        """Analyze risks for a task and path combination.

        Args:
            task: The task
            path: The execution path

        Returns:
            Risk analysis dictionary
        """
        analysis = {
            "task_risk": task.risk_level.value,
            "path_risk": path.risk_level,
            "overall_risk": "moderate",
            "mitigations": [],
            "concerns": []
        }

        # Calculate combined risk
        task_risk_values = {
            RiskLevel.MINIMAL: 0.1,
            RiskLevel.LOW: 0.3,
            RiskLevel.MODERATE: 0.5,
            RiskLevel.HIGH: 0.7,
            RiskLevel.CRITICAL: 0.9
        }
        combined_risk = (
            task_risk_values.get(task.risk_level, 0.5) + path.risk_level
        ) / 2

        if combined_risk < 0.3:
            analysis["overall_risk"] = "low"
        elif combined_risk < 0.6:
            analysis["overall_risk"] = "moderate"
        else:
            analysis["overall_risk"] = "high"

        # Suggest mitigations based on strategy
        if path.strategy == StrategyType.AGGRESSIVE:
            analysis["mitigations"].append("Create checkpoint before execution")
            analysis["mitigations"].append("Have rollback plan ready")
            analysis["concerns"].append("Fast execution may miss edge cases")
        elif path.strategy == StrategyType.EXPLORATORY:
            analysis["mitigations"].append("Set time limit for exploration")
            analysis["concerns"].append("May consume extra tokens")

        if task.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            analysis["mitigations"].append("Run in isolated environment")
            analysis["mitigations"].append("Enable verbose logging")

        return analysis

    def _allocate_resources(
        self,
        task: Task,
        path: ExecutionPath
    ) -> Dict[str, int]:
        """Allocate resources for task execution.

        Args:
            task: The task
            path: The execution path

        Returns:
            Resource allocation dictionary
        """
        available_tokens = self.token_budget - self.tokens_used

        # Base allocation from path estimate
        token_allocation = path.estimated_tokens

        # Adjust based on priority
        priority_multipliers = {
            Priority.CRITICAL: 1.5,
            Priority.HIGH: 1.2,
            Priority.MEDIUM: 1.0,
            Priority.LOW: 0.8,
            Priority.OPTIONAL: 0.6
        }
        token_allocation = int(
            token_allocation * priority_multipliers.get(task.priority, 1.0)
        )

        # Don't exceed available tokens
        token_allocation = min(token_allocation, available_tokens)

        return {
            "tokens": token_allocation,
            "iterations": self._estimate_iterations(task, path),
            "checkpoint_interval": 5 if task.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL] else 10
        }

    def _estimate_iterations(self, task: Task, path: ExecutionPath) -> int:
        """Estimate number of iterations needed.

        Args:
            task: The task
            path: The execution path

        Returns:
            Estimated iteration count
        """
        base = task.estimated_effort

        strategy_multipliers = {
            StrategyType.AGGRESSIVE: 0.7,
            StrategyType.CONSERVATIVE: 1.5,
            StrategyType.BALANCED: 1.0,
            StrategyType.EXPLORATORY: 1.3,
            StrategyType.EXPLOIT: 0.8
        }

        multiplier = strategy_multipliers.get(path.strategy, 1.0)
        return max(1, int(base * multiplier))

    def _generate_ceo_reasoning(
        self,
        task: Task,
        result: Any
    ) -> str:
        """Generate CEO-level reasoning for the decision.

        Args:
            task: The task
            result: Simulation result

        Returns:
            Reasoning string
        """
        reasons = []

        # Apply CEO principles
        reasons.append(f"Strategy: {result.path.strategy.value.title()}")

        # Success probability assessment
        if result.mean_success_rate >= 0.8:
            reasons.append("High confidence in success - proceed decisively")
        elif result.mean_success_rate >= 0.6:
            reasons.append("Good probability of success - worth the investment")
        else:
            reasons.append("Lower confidence - will iterate and adjust as needed")

        # Token efficiency
        if result.token_efficiency_score >= 0.5:
            reasons.append("Efficient use of resources")
        else:
            reasons.append("Higher resource investment - justified by priority")

        # Risk consideration
        if task.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            reasons.append("Taking measured approach due to risk level")

        # Priority consideration
        if task.priority == Priority.CRITICAL:
            reasons.append("Critical priority - allocating maximum focus")

        return " | ".join(reasons)

    def _predict_outcome(self, task: Task, result: Any) -> str:
        """Predict expected outcome.

        Args:
            task: The task
            result: Simulation result

        Returns:
            Expected outcome description
        """
        confidence = result.mean_success_rate

        if confidence >= 0.8:
            return f"Expect successful completion with {confidence:.0%} confidence"
        elif confidence >= 0.6:
            return f"Good chance of success ({confidence:.0%}), may need minor adjustments"
        else:
            return f"May require iteration ({confidence:.0%}), prepared to adapt strategy"

    def complete_task(self, task: Task, success: bool, tokens_used: int) -> None:
        """Mark a task as complete and record outcome.

        Args:
            task: The completed task
            success: Whether it was successful
            tokens_used: Tokens consumed
        """
        # Update tokens
        self.tokens_used += tokens_used

        # Move to completed
        if task in self.task_queue:
            self.task_queue.remove(task)
        self.completed_tasks.append(task)

        # Find the decision for this task
        for decision in self.decision_history:
            if decision.task.id == task.id:
                # Record outcome in decision engine
                self.decision_engine.record_outcome(
                    path=decision.recommended_path,
                    success=success,
                    tokens_used=tokens_used,
                    notes=f"Task: {task.description[:50]}"
                )
                break

        # Update metrics
        if "completed" not in self.performance_metrics:
            self.performance_metrics["completed"] = 0
            self.performance_metrics["successful"] = 0
            self.performance_metrics["total_tokens"] = 0

        self.performance_metrics["completed"] += 1
        if success:
            self.performance_metrics["successful"] += 1
        self.performance_metrics["total_tokens"] += tokens_used

        # Save memory
        self._save_memory()

        logger.info(f"Task {task.id} completed: {'success' if success else 'failure'} "
                   f"({tokens_used} tokens)")

    def get_status(self) -> Dict[str, Any]:
        """Get current agent status.

        Returns:
            Status dictionary
        """
        return {
            "name": self.name,
            "token_budget": self.token_budget,
            "tokens_used": self.tokens_used,
            "tokens_remaining": self.token_budget - self.tokens_used,
            "tasks_pending": len(self.task_queue),
            "tasks_completed": len(self.completed_tasks),
            "decisions_made": len(self.decision_history),
            "success_rate": (
                self.performance_metrics.get("successful", 0) /
                max(1, self.performance_metrics.get("completed", 1))
            ),
            "risk_tolerance": self.risk_tolerance,
            "decision_engine_stats": self.decision_engine.get_statistics()
        }

    def get_executive_summary(self) -> str:
        """Generate an executive summary of current state.

        Returns:
            Executive summary string
        """
        status = self.get_status()

        summary_parts = [
            f"=== {self.name} Executive Summary ===",
            f"",
            f"Resource Status:",
            f"  - Token Budget: {status['tokens_remaining']:,} / {status['token_budget']:,} remaining ({100 * status['tokens_remaining'] / status['token_budget']:.1f}%)",
            f"",
            f"Task Status:",
            f"  - Pending: {status['tasks_pending']}",
            f"  - Completed: {status['tasks_completed']}",
            f"  - Success Rate: {status['success_rate']:.1%}",
            f"",
            f"Decision Metrics:",
            f"  - Decisions Made: {status['decisions_made']}",
        ]

        if self.task_queue:
            next_task = self.get_next_task()
            if next_task:
                summary_parts.extend([
                    f"",
                    f"Next Up:",
                    f"  - Task: {next_task.description[:60]}...",
                    f"  - Priority: {next_task.priority.value}",
                    f"  - Est. Effort: {next_task.estimated_effort}/10",
                ])

        return "\n".join(summary_parts)


# Singleton instance for easy access
_default_agent: Optional[StrategicAgent] = None


def get_strategic_agent(
    name: str = "RalphCEO",
    token_budget: int = 100000,
    reset: bool = False
) -> StrategicAgent:
    """Get or create the default strategic agent.

    Args:
        name: Agent name
        token_budget: Token budget
        reset: Force create new agent

    Returns:
        Strategic agent instance
    """
    global _default_agent

    if _default_agent is None or reset:
        _default_agent = StrategicAgent(
            name=name,
            token_budget=token_budget
        )

    return _default_agent
