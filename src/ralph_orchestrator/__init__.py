# ABOUTME: Ralph Orchestrator package for AI agent orchestration
# ABOUTME: Implements the Ralph Wiggum technique with multi-tool support
# ABOUTME: Enhanced with Monte Carlo decision engine and BMAD agent roles

"""Ralph Orchestrator - AI agent orchestration with strategic decision-making.

This package provides:
- RalphOrchestrator: Core orchestration loop for AI agents
- MonteCarloEngine: Decision optimization via simulation
- StrategicAgent: CEO-level strategic decision making
- MemoryStore: Persistent learning and pattern recognition
- BMDAAgentRegistry: Specialized agent roles (BMAD methodology)
"""

__version__ = "1.3.0"

from .orchestrator import RalphOrchestrator
from .metrics import Metrics, CostTracker, IterationStats, TriggerReason
from .error_formatter import ClaudeErrorFormatter, ErrorMessage
from .verbose_logger import VerboseLogger
from .output import DiffStats, DiffFormatter, RalphConsole

# New modules - strategic decision making
from .decision_engine import (
    MonteCarloEngine,
    DecisionContext,
    ExecutionPath,
    StrategyType,
    ConfidenceTracker,
    SimulationResult,
)
from .strategic_agent import (
    StrategicAgent,
    Task,
    Priority,
    RiskLevel,
    StrategicDecision,
    CEOPrinciples,
    get_strategic_agent,
)
from .memory import (
    MemoryStore,
    Memory,
    Pattern,
    MemoryType,
    get_memory_store,
)
from .bmad_agents import (
    AgentRole,
    AgentProfile,
    WorkflowPhase,
    TaskRouter,
    BMDAAgentRegistry,
    BMDAWorkflowManager,
    get_agent_for_task,
    create_workflow,
)

__all__ = [
    # Core orchestration
    "RalphOrchestrator",
    "Metrics",
    "CostTracker",
    "IterationStats",
    "TriggerReason",
    # Error handling
    "ClaudeErrorFormatter",
    "ErrorMessage",
    # Logging and output
    "VerboseLogger",
    "DiffStats",
    "DiffFormatter",
    "RalphConsole",
    # Monte Carlo decision engine
    "MonteCarloEngine",
    "DecisionContext",
    "ExecutionPath",
    "StrategyType",
    "ConfidenceTracker",
    "SimulationResult",
    # Strategic agent
    "StrategicAgent",
    "Task",
    "Priority",
    "RiskLevel",
    "StrategicDecision",
    "CEOPrinciples",
    "get_strategic_agent",
    # Memory system
    "MemoryStore",
    "Memory",
    "Pattern",
    "MemoryType",
    "get_memory_store",
    # BMAD agents
    "AgentRole",
    "AgentProfile",
    "WorkflowPhase",
    "TaskRouter",
    "BMDAAgentRegistry",
    "BMDAWorkflowManager",
    "get_agent_for_task",
    "create_workflow",
]
