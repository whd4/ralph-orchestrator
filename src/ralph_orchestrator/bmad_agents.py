# ABOUTME: BMAD Agent roles and workflow routing integration
# ABOUTME: Implements specialized agents based on BMAD methodology

"""BMAD Agent Roles for Ralph Orchestrator.

This module implements specialized agent roles based on the BMAD
(Build, Measure, Analyze, Deploy) methodology, providing focused
capabilities for different phases of software development.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class AgentRole(str, Enum):
    """BMAD-inspired agent roles."""
    # Research & Analysis
    ANALYST = "analyst"                 # Market research, requirements analysis
    RESEARCHER = "researcher"           # Technical research, documentation review

    # Design & Planning
    ARCHITECT = "architect"             # System design, technical decisions
    PRODUCT_MANAGER = "product_manager" # Product strategy, roadmap
    SCRUM_MASTER = "scrum_master"       # Sprint planning, agile processes

    # Development
    DEVELOPER = "developer"             # Code implementation
    FRONTEND_DEV = "frontend_dev"       # UI/UX implementation
    BACKEND_DEV = "backend_dev"         # Server-side implementation
    DATA_ENGINEER = "data_engineer"     # Data pipelines, databases

    # Quality & Testing
    QA_ENGINEER = "qa_engineer"         # Testing, quality assurance
    SECURITY_ANALYST = "security_analyst"  # Security review, vulnerability assessment

    # Operations
    DEVOPS = "devops"                   # CI/CD, infrastructure
    TECH_LEAD = "tech_lead"             # Code review, mentoring

    # Documentation
    TECH_WRITER = "tech_writer"         # Documentation, guides
    API_DESIGNER = "api_designer"       # API design, OpenAPI specs


class WorkflowPhase(str, Enum):
    """Phases in the BMAD workflow."""
    BUILD = "build"       # Development phase
    MEASURE = "measure"   # Testing & metrics
    ANALYZE = "analyze"   # Review & analysis
    DEPLOY = "deploy"     # Release & operations


@dataclass
class AgentCapability:
    """Defines what an agent can do."""
    name: str
    description: str
    skills: List[str]
    tools: List[str]
    input_types: List[str]
    output_types: List[str]


@dataclass
class AgentProfile:
    """Complete profile for a BMAD agent."""
    role: AgentRole
    name: str
    description: str
    phase: WorkflowPhase
    capabilities: List[AgentCapability]
    expertise_areas: List[str]
    default_tools: List[str]
    system_prompt: str
    constraints: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BMDAAgentRegistry:
    """Registry of BMAD agent profiles and capabilities."""

    # Pre-defined agent profiles
    AGENT_PROFILES: Dict[AgentRole, AgentProfile] = {}

    @classmethod
    def _init_profiles(cls) -> None:
        """Initialize agent profiles."""
        if cls.AGENT_PROFILES:
            return

        cls.AGENT_PROFILES = {
            AgentRole.ANALYST: AgentProfile(
                role=AgentRole.ANALYST,
                name="Business Analyst",
                description="Gathers requirements, analyzes market, identifies opportunities",
                phase=WorkflowPhase.ANALYZE,
                capabilities=[
                    AgentCapability(
                        name="market_research",
                        description="Research market trends and competitors",
                        skills=["research", "analysis", "synthesis"],
                        tools=["WebSearch", "Read"],
                        input_types=["text", "url"],
                        output_types=["report", "markdown"]
                    ),
                    AgentCapability(
                        name="requirements_gathering",
                        description="Extract and document requirements",
                        skills=["interviewing", "documentation", "prioritization"],
                        tools=["Read", "Write"],
                        input_types=["text", "conversation"],
                        output_types=["requirements_doc", "user_stories"]
                    )
                ],
                expertise_areas=["market analysis", "requirements", "user research"],
                default_tools=["WebSearch", "Read", "Write"],
                system_prompt="""You are a Business Analyst. Your role is to:
- Gather and analyze requirements
- Research market trends and competitors
- Identify opportunities and risks
- Create clear requirement documents
- Prioritize features based on business value

Always validate assumptions with research. Focus on user needs and business outcomes."""
            ),

            AgentRole.ARCHITECT: AgentProfile(
                role=AgentRole.ARCHITECT,
                name="System Architect",
                description="Designs system architecture, makes technical decisions",
                phase=WorkflowPhase.BUILD,
                capabilities=[
                    AgentCapability(
                        name="system_design",
                        description="Create system architecture diagrams and documentation",
                        skills=["design", "documentation", "decision-making"],
                        tools=["Read", "Write", "Grep", "Glob"],
                        input_types=["requirements", "codebase"],
                        output_types=["architecture_doc", "diagrams", "adr"]
                    ),
                    AgentCapability(
                        name="tech_stack_selection",
                        description="Evaluate and recommend technology choices",
                        skills=["research", "evaluation", "comparison"],
                        tools=["WebSearch", "Read"],
                        input_types=["requirements", "constraints"],
                        output_types=["tech_recommendation", "comparison_matrix"]
                    )
                ],
                expertise_areas=["system design", "scalability", "patterns"],
                default_tools=["Read", "Write", "Glob", "Grep"],
                system_prompt="""You are a System Architect. Your role is to:
- Design scalable, maintainable system architectures
- Make and document technical decisions (ADRs)
- Evaluate technology choices
- Ensure designs meet non-functional requirements
- Create clear technical documentation

Think about scalability, security, and maintainability. Document trade-offs clearly."""
            ),

            AgentRole.DEVELOPER: AgentProfile(
                role=AgentRole.DEVELOPER,
                name="Software Developer",
                description="Implements features, writes code, follows TDD",
                phase=WorkflowPhase.BUILD,
                capabilities=[
                    AgentCapability(
                        name="code_implementation",
                        description="Write clean, tested code",
                        skills=["coding", "testing", "debugging"],
                        tools=["Read", "Write", "Edit", "Bash"],
                        input_types=["requirements", "design_doc", "test_cases"],
                        output_types=["code", "tests", "documentation"]
                    ),
                    AgentCapability(
                        name="refactoring",
                        description="Improve code quality without changing behavior",
                        skills=["refactoring", "code_review", "optimization"],
                        tools=["Read", "Edit", "Glob", "Grep"],
                        input_types=["code"],
                        output_types=["code", "documentation"]
                    )
                ],
                expertise_areas=["implementation", "TDD", "clean code"],
                default_tools=["Read", "Write", "Edit", "Bash"],
                system_prompt="""You are a Software Developer. Your role is to:
- Write clean, maintainable code
- Follow TDD: write tests first, then implement
- Document your code appropriately
- Handle errors gracefully
- Commit changes with clear messages

Focus on quality over speed. Test your code thoroughly."""
            ),

            AgentRole.QA_ENGINEER: AgentProfile(
                role=AgentRole.QA_ENGINEER,
                name="QA Engineer",
                description="Tests software, ensures quality, finds bugs",
                phase=WorkflowPhase.MEASURE,
                capabilities=[
                    AgentCapability(
                        name="test_planning",
                        description="Create comprehensive test plans",
                        skills=["test_design", "risk_assessment", "planning"],
                        tools=["Read", "Write"],
                        input_types=["requirements", "code"],
                        output_types=["test_plan", "test_cases"]
                    ),
                    AgentCapability(
                        name="test_execution",
                        description="Execute tests and report results",
                        skills=["testing", "debugging", "reporting"],
                        tools=["Bash", "Read", "Write"],
                        input_types=["test_cases", "code"],
                        output_types=["test_results", "bug_reports"]
                    )
                ],
                expertise_areas=["testing", "quality assurance", "automation"],
                default_tools=["Bash", "Read", "Write", "Grep"],
                system_prompt="""You are a QA Engineer. Your role is to:
- Create comprehensive test plans
- Write and execute tests
- Find and document bugs clearly
- Verify fixes and regressions
- Ensure quality standards are met

Think like a user and an attacker. Test edge cases thoroughly."""
            ),

            AgentRole.DEVOPS: AgentProfile(
                role=AgentRole.DEVOPS,
                name="DevOps Engineer",
                description="Manages CI/CD, infrastructure, deployments",
                phase=WorkflowPhase.DEPLOY,
                capabilities=[
                    AgentCapability(
                        name="ci_cd_setup",
                        description="Configure CI/CD pipelines",
                        skills=["automation", "scripting", "configuration"],
                        tools=["Read", "Write", "Bash"],
                        input_types=["requirements", "code"],
                        output_types=["pipeline_config", "scripts"]
                    ),
                    AgentCapability(
                        name="deployment",
                        description="Deploy and monitor applications",
                        skills=["deployment", "monitoring", "troubleshooting"],
                        tools=["Bash", "Read", "Write"],
                        input_types=["code", "config"],
                        output_types=["deployment_logs", "monitoring_config"]
                    )
                ],
                expertise_areas=["CI/CD", "infrastructure", "monitoring"],
                default_tools=["Bash", "Read", "Write", "Glob"],
                system_prompt="""You are a DevOps Engineer. Your role is to:
- Set up and maintain CI/CD pipelines
- Automate deployments and infrastructure
- Monitor system health and performance
- Ensure security and reliability
- Document operational procedures

Automate everything possible. Think about failure scenarios."""
            ),

            AgentRole.PRODUCT_MANAGER: AgentProfile(
                role=AgentRole.PRODUCT_MANAGER,
                name="Product Manager",
                description="Defines product vision, prioritizes features",
                phase=WorkflowPhase.ANALYZE,
                capabilities=[
                    AgentCapability(
                        name="roadmap_planning",
                        description="Create and maintain product roadmap",
                        skills=["planning", "prioritization", "communication"],
                        tools=["Read", "Write"],
                        input_types=["requirements", "feedback"],
                        output_types=["roadmap", "priorities"]
                    )
                ],
                expertise_areas=["product strategy", "prioritization", "stakeholder management"],
                default_tools=["Read", "Write", "WebSearch"],
                system_prompt="""You are a Product Manager. Your role is to:
- Define and communicate product vision
- Prioritize features based on value and effort
- Gather and synthesize user feedback
- Create clear product roadmaps
- Balance stakeholder needs

Focus on delivering user value. Make data-driven decisions."""
            ),

            AgentRole.SCRUM_MASTER: AgentProfile(
                role=AgentRole.SCRUM_MASTER,
                name="Scrum Master",
                description="Facilitates agile processes, removes blockers",
                phase=WorkflowPhase.BUILD,
                capabilities=[
                    AgentCapability(
                        name="sprint_planning",
                        description="Plan and facilitate sprints",
                        skills=["facilitation", "planning", "coaching"],
                        tools=["Read", "Write"],
                        input_types=["backlog", "capacity"],
                        output_types=["sprint_plan", "burndown"]
                    )
                ],
                expertise_areas=["agile", "facilitation", "team dynamics"],
                default_tools=["Read", "Write"],
                system_prompt="""You are a Scrum Master. Your role is to:
- Facilitate sprint ceremonies
- Remove blockers and impediments
- Coach team on agile practices
- Protect team from distractions
- Track and communicate progress

Enable the team to do their best work. Focus on continuous improvement."""
            ),

            AgentRole.TECH_WRITER: AgentProfile(
                role=AgentRole.TECH_WRITER,
                name="Technical Writer",
                description="Creates documentation, guides, and tutorials",
                phase=WorkflowPhase.BUILD,
                capabilities=[
                    AgentCapability(
                        name="documentation",
                        description="Write clear technical documentation",
                        skills=["writing", "research", "organization"],
                        tools=["Read", "Write", "Glob"],
                        input_types=["code", "notes"],
                        output_types=["documentation", "tutorials", "guides"]
                    )
                ],
                expertise_areas=["technical writing", "documentation", "user guides"],
                default_tools=["Read", "Write", "Glob"],
                system_prompt="""You are a Technical Writer. Your role is to:
- Create clear, accurate documentation
- Write user guides and tutorials
- Document APIs and interfaces
- Keep documentation up to date
- Make complex topics accessible

Write for your audience. Be clear and concise."""
            ),

            AgentRole.SECURITY_ANALYST: AgentProfile(
                role=AgentRole.SECURITY_ANALYST,
                name="Security Analyst",
                description="Reviews security, identifies vulnerabilities",
                phase=WorkflowPhase.MEASURE,
                capabilities=[
                    AgentCapability(
                        name="security_review",
                        description="Review code and architecture for security issues",
                        skills=["security", "analysis", "threat_modeling"],
                        tools=["Read", "Grep", "Glob"],
                        input_types=["code", "architecture"],
                        output_types=["security_report", "recommendations"]
                    )
                ],
                expertise_areas=["security", "OWASP", "threat modeling"],
                default_tools=["Read", "Grep", "Glob"],
                system_prompt="""You are a Security Analyst. Your role is to:
- Review code for security vulnerabilities
- Identify OWASP Top 10 issues
- Perform threat modeling
- Recommend security improvements
- Verify security fixes

Think like an attacker. Prioritize by risk."""
            )
        }

    @classmethod
    def get_profile(cls, role: AgentRole) -> Optional[AgentProfile]:
        """Get profile for a role.

        Args:
            role: The agent role

        Returns:
            Agent profile or None
        """
        cls._init_profiles()
        return cls.AGENT_PROFILES.get(role)

    @classmethod
    def get_all_profiles(cls) -> Dict[AgentRole, AgentProfile]:
        """Get all agent profiles.

        Returns:
            Dictionary of all profiles
        """
        cls._init_profiles()
        return cls.AGENT_PROFILES.copy()

    @classmethod
    def get_roles_for_phase(cls, phase: WorkflowPhase) -> List[AgentRole]:
        """Get roles relevant to a workflow phase.

        Args:
            phase: The workflow phase

        Returns:
            List of relevant roles
        """
        cls._init_profiles()
        return [
            role for role, profile in cls.AGENT_PROFILES.items()
            if profile.phase == phase
        ]


class TaskRouter:
    """Routes tasks to appropriate BMAD agents."""

    # Keywords to role mapping
    ROLE_KEYWORDS: Dict[AgentRole, List[str]] = {
        AgentRole.ANALYST: [
            "research", "market", "analyze", "requirements", "user needs",
            "competitor", "opportunity", "business case"
        ],
        AgentRole.ARCHITECT: [
            "design", "architecture", "system", "scalability", "pattern",
            "decision", "tech stack", "infrastructure"
        ],
        AgentRole.DEVELOPER: [
            "implement", "code", "develop", "build", "create", "function",
            "feature", "fix", "refactor"
        ],
        AgentRole.QA_ENGINEER: [
            "test", "quality", "bug", "verify", "validate", "coverage",
            "regression", "edge case"
        ],
        AgentRole.DEVOPS: [
            "deploy", "ci/cd", "pipeline", "docker", "kubernetes", "monitor",
            "infrastructure", "automation"
        ],
        AgentRole.SECURITY_ANALYST: [
            "security", "vulnerability", "owasp", "auth", "encryption",
            "threat", "penetration", "audit"
        ],
        AgentRole.TECH_WRITER: [
            "document", "readme", "guide", "tutorial", "api doc",
            "changelog", "specification"
        ],
        AgentRole.PRODUCT_MANAGER: [
            "roadmap", "priority", "feature", "stakeholder", "backlog",
            "vision", "strategy"
        ],
        AgentRole.SCRUM_MASTER: [
            "sprint", "agile", "standup", "retrospective", "velocity",
            "blocker", "planning"
        ]
    }

    def __init__(self):
        """Initialize the task router."""
        self._registry = BMDAAgentRegistry()

    def route_task(self, task_description: str) -> Tuple[AgentRole, float]:
        """Route a task to the most appropriate agent.

        Args:
            task_description: Description of the task

        Returns:
            Tuple of (recommended role, confidence score)
        """
        task_lower = task_description.lower()
        scores: Dict[AgentRole, int] = {role: 0 for role in AgentRole}

        # Score each role based on keyword matches
        for role, keywords in self.ROLE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in task_lower:
                    scores[role] += 1

        # Find best match
        best_role = max(scores, key=scores.get)
        best_score = scores[best_role]

        # Calculate confidence
        total_keywords = sum(len(kw) for kw in self.ROLE_KEYWORDS.values())
        max_possible = len(self.ROLE_KEYWORDS.get(best_role, []))
        confidence = best_score / max_possible if max_possible > 0 else 0.5

        # Default to developer if no clear match
        if best_score == 0:
            return AgentRole.DEVELOPER, 0.5

        return best_role, min(confidence, 1.0)

    def get_workflow(self, task_description: str) -> List[AgentRole]:
        """Get recommended workflow for a complex task.

        Args:
            task_description: Description of the task

        Returns:
            List of agents in recommended execution order
        """
        task_lower = task_description.lower()
        workflow = []

        # Determine if this is a full project or specific task
        is_full_project = any(
            word in task_lower for word in
            ["project", "application", "system", "platform", "product"]
        )

        if is_full_project:
            # Full development workflow
            workflow = [
                AgentRole.ANALYST,
                AgentRole.PRODUCT_MANAGER,
                AgentRole.ARCHITECT,
                AgentRole.DEVELOPER,
                AgentRole.QA_ENGINEER,
                AgentRole.SECURITY_ANALYST,
                AgentRole.TECH_WRITER,
                AgentRole.DEVOPS
            ]
        else:
            # Single task - route to primary agent
            primary_role, _ = self.route_task(task_description)
            workflow = [primary_role]

            # Add QA for implementation tasks
            if primary_role == AgentRole.DEVELOPER:
                workflow.append(AgentRole.QA_ENGINEER)

        return workflow


@dataclass
class WorkflowExecution:
    """Tracks execution of a BMAD workflow."""
    id: str
    task_description: str
    workflow: List[AgentRole]
    current_step: int = 0
    status: str = "pending"
    results: Dict[str, Any] = field(default_factory=dict)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def advance(self) -> Optional[AgentRole]:
        """Advance to next step in workflow.

        Returns:
            Next agent role or None if complete
        """
        if self.current_step >= len(self.workflow):
            self.status = "completed"
            self.completed_at = datetime.now()
            return None

        role = self.workflow[self.current_step]
        self.current_step += 1

        if self.started_at is None:
            self.started_at = datetime.now()

        return role

    def record_result(self, role: AgentRole, result: Any) -> None:
        """Record result for a workflow step.

        Args:
            role: The agent role that produced the result
            result: The result to record
        """
        self.results[role.value] = {
            "result": result,
            "timestamp": datetime.now().isoformat()
        }


class BMDAWorkflowManager:
    """Manages BMAD workflow execution."""

    def __init__(self):
        """Initialize workflow manager."""
        self.router = TaskRouter()
        self.active_workflows: Dict[str, WorkflowExecution] = {}

    def create_workflow(self, task_description: str) -> WorkflowExecution:
        """Create a new workflow for a task.

        Args:
            task_description: Description of the task

        Returns:
            New workflow execution object
        """
        workflow = self.router.get_workflow(task_description)

        execution = WorkflowExecution(
            id=f"wf_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            task_description=task_description,
            workflow=workflow
        )

        self.active_workflows[execution.id] = execution
        logger.info(f"Created workflow {execution.id} with {len(workflow)} steps")

        return execution

    def get_next_agent(self, workflow_id: str) -> Optional[AgentProfile]:
        """Get the next agent profile for a workflow.

        Args:
            workflow_id: ID of the workflow

        Returns:
            Next agent profile or None
        """
        if workflow_id not in self.active_workflows:
            return None

        execution = self.active_workflows[workflow_id]
        next_role = execution.advance()

        if next_role is None:
            return None

        return BMDAAgentRegistry.get_profile(next_role)

    def get_agent_prompt(
        self,
        profile: AgentProfile,
        task: str,
        context: Optional[str] = None
    ) -> str:
        """Generate a prompt for an agent.

        Args:
            profile: The agent profile
            task: The specific task
            context: Optional context from previous steps

        Returns:
            Formatted prompt string
        """
        prompt_parts = [
            f"# Role: {profile.name}",
            f"",
            profile.system_prompt,
            f"",
            f"# Current Task",
            task,
        ]

        if context:
            prompt_parts.extend([
                f"",
                f"# Context from Previous Steps",
                context
            ])

        if profile.constraints:
            prompt_parts.extend([
                f"",
                f"# Constraints",
                *[f"- {c}" for c in profile.constraints]
            ])

        return "\n".join(prompt_parts)


# Convenience functions
def get_agent_for_task(task_description: str) -> Tuple[AgentProfile, float]:
    """Get the best agent for a task.

    Args:
        task_description: Description of the task

    Returns:
        Tuple of (agent profile, confidence)
    """
    router = TaskRouter()
    role, confidence = router.route_task(task_description)
    profile = BMDAAgentRegistry.get_profile(role)
    return profile, confidence


def create_workflow(task_description: str) -> WorkflowExecution:
    """Create a workflow for a task.

    Args:
        task_description: Description of the task

    Returns:
        Workflow execution object
    """
    manager = BMDAWorkflowManager()
    return manager.create_workflow(task_description)
