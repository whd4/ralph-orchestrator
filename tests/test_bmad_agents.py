# ABOUTME: Tests for BMAD Agent Roles and Workflow Routing
# ABOUTME: Verifies agent profiles and task routing

"""Tests for BMAD Agent Roles."""

import pytest

from ralph_orchestrator.bmad_agents import (
    AgentRole,
    AgentProfile,
    WorkflowPhase,
    TaskRouter,
    BMDAAgentRegistry,
    BMDAWorkflowManager,
    WorkflowExecution,
    get_agent_for_task,
    create_workflow,
)


class TestAgentRole:
    """Tests for AgentRole enum."""

    def test_all_roles_exist(self):
        """Test all expected roles exist."""
        roles = [
            AgentRole.ANALYST,
            AgentRole.ARCHITECT,
            AgentRole.DEVELOPER,
            AgentRole.QA_ENGINEER,
            AgentRole.DEVOPS,
            AgentRole.PRODUCT_MANAGER,
            AgentRole.SCRUM_MASTER,
            AgentRole.TECH_WRITER,
            AgentRole.SECURITY_ANALYST
        ]

        assert len(roles) >= 9

    def test_role_values(self):
        """Test role string values."""
        assert AgentRole.ANALYST.value == "analyst"
        assert AgentRole.DEVELOPER.value == "developer"
        assert AgentRole.QA_ENGINEER.value == "qa_engineer"


class TestWorkflowPhase:
    """Tests for WorkflowPhase enum."""

    def test_phases_exist(self):
        """Test all workflow phases exist."""
        phases = [
            WorkflowPhase.BUILD,
            WorkflowPhase.MEASURE,
            WorkflowPhase.ANALYZE,
            WorkflowPhase.DEPLOY
        ]

        assert len(phases) == 4

    def test_phase_values(self):
        """Test phase string values."""
        assert WorkflowPhase.BUILD.value == "build"
        assert WorkflowPhase.MEASURE.value == "measure"


class TestBMDAAgentRegistry:
    """Tests for BMDAAgentRegistry class."""

    def test_get_profile(self):
        """Test getting an agent profile."""
        profile = BMDAAgentRegistry.get_profile(AgentRole.DEVELOPER)

        assert profile is not None
        assert isinstance(profile, AgentProfile)
        assert profile.role == AgentRole.DEVELOPER

    def test_profile_has_system_prompt(self):
        """Test profiles have system prompts."""
        profile = BMDAAgentRegistry.get_profile(AgentRole.ARCHITECT)

        assert profile.system_prompt != ""
        assert len(profile.system_prompt) > 50

    def test_profile_has_capabilities(self):
        """Test profiles have capabilities."""
        profile = BMDAAgentRegistry.get_profile(AgentRole.QA_ENGINEER)

        assert len(profile.capabilities) > 0
        assert all(cap.name != "" for cap in profile.capabilities)

    def test_get_all_profiles(self):
        """Test getting all profiles."""
        profiles = BMDAAgentRegistry.get_all_profiles()

        assert isinstance(profiles, dict)
        assert len(profiles) >= 9

    def test_get_roles_for_phase(self):
        """Test getting roles for a phase."""
        build_roles = BMDAAgentRegistry.get_roles_for_phase(WorkflowPhase.BUILD)

        assert isinstance(build_roles, list)
        assert AgentRole.DEVELOPER in build_roles


class TestTaskRouter:
    """Tests for TaskRouter class."""

    @pytest.fixture
    def router(self):
        """Create task router."""
        return TaskRouter()

    def test_route_implementation_task(self, router):
        """Test routing implementation task."""
        role, confidence = router.route_task("Implement user authentication feature")

        assert role == AgentRole.DEVELOPER
        assert confidence > 0

    def test_route_testing_task(self, router):
        """Test routing testing task."""
        role, confidence = router.route_task("Write unit tests for the API")

        assert role == AgentRole.QA_ENGINEER
        assert confidence > 0

    def test_route_architecture_task(self, router):
        """Test routing architecture task."""
        role, confidence = router.route_task("Design the system architecture")

        assert role == AgentRole.ARCHITECT
        assert confidence > 0

    def test_route_deployment_task(self, router):
        """Test routing deployment task."""
        role, confidence = router.route_task("Set up CI/CD pipeline")

        assert role == AgentRole.DEVOPS
        assert confidence > 0

    def test_route_security_task(self, router):
        """Test routing security task."""
        role, confidence = router.route_task("Perform security audit and threat modeling analysis")

        assert role == AgentRole.SECURITY_ANALYST
        assert confidence > 0

    def test_route_unknown_task(self, router):
        """Test routing unknown task defaults to developer."""
        role, confidence = router.route_task("Do something completely random xyz123")

        # Should default to developer
        assert role == AgentRole.DEVELOPER
        assert confidence == 0.5

    def test_get_workflow_for_project(self, router):
        """Test getting workflow for a full project."""
        workflow = router.get_workflow("Build a complete e-commerce application")

        assert isinstance(workflow, list)
        assert len(workflow) > 1  # Multiple agents for project
        assert AgentRole.ANALYST in workflow
        assert AgentRole.DEVELOPER in workflow

    def test_get_workflow_for_single_task(self, router):
        """Test getting workflow for single task."""
        workflow = router.get_workflow("Fix the login bug")

        assert isinstance(workflow, list)
        assert len(workflow) >= 1


class TestWorkflowExecution:
    """Tests for WorkflowExecution class."""

    def test_execution_creation(self):
        """Test creating workflow execution."""
        execution = WorkflowExecution(
            id="wf_001",
            task_description="Build an API",
            workflow=[AgentRole.ARCHITECT, AgentRole.DEVELOPER, AgentRole.QA_ENGINEER]
        )

        assert execution.id == "wf_001"
        assert len(execution.workflow) == 3
        assert execution.current_step == 0

    def test_advance_workflow(self):
        """Test advancing through workflow."""
        execution = WorkflowExecution(
            id="wf_002",
            task_description="Test task",
            workflow=[AgentRole.DEVELOPER, AgentRole.QA_ENGINEER]
        )

        role1 = execution.advance()
        assert role1 == AgentRole.DEVELOPER
        assert execution.current_step == 1

        role2 = execution.advance()
        assert role2 == AgentRole.QA_ENGINEER
        assert execution.current_step == 2

        role3 = execution.advance()
        assert role3 is None  # Completed
        assert execution.status == "completed"

    def test_record_result(self):
        """Test recording step results."""
        execution = WorkflowExecution(
            id="wf_003",
            task_description="Test",
            workflow=[AgentRole.DEVELOPER]
        )

        execution.record_result(AgentRole.DEVELOPER, {"output": "Code implemented"})

        assert AgentRole.DEVELOPER.value in execution.results
        assert "output" in execution.results[AgentRole.DEVELOPER.value]["result"]


class TestBMDAWorkflowManager:
    """Tests for BMDAWorkflowManager class."""

    @pytest.fixture
    def manager(self):
        """Create workflow manager."""
        return BMDAWorkflowManager()

    def test_create_workflow(self, manager):
        """Test creating a workflow."""
        execution = manager.create_workflow("Implement user dashboard")

        assert isinstance(execution, WorkflowExecution)
        assert len(execution.workflow) > 0

    def test_get_next_agent(self, manager):
        """Test getting next agent in workflow."""
        execution = manager.create_workflow("Build REST API")
        workflow_id = execution.id

        profile = manager.get_next_agent(workflow_id)

        assert profile is not None or len(execution.workflow) == 0
        if profile:
            assert isinstance(profile, AgentProfile)

    def test_get_agent_prompt(self, manager):
        """Test generating agent prompt."""
        profile = BMDAAgentRegistry.get_profile(AgentRole.DEVELOPER)

        prompt = manager.get_agent_prompt(
            profile=profile,
            task="Implement login form",
            context="Previous agent designed the API"
        )

        assert "Role:" in prompt
        assert "login form" in prompt.lower()


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_get_agent_for_task(self):
        """Test get_agent_for_task function."""
        profile, confidence = get_agent_for_task("Write unit tests")

        assert isinstance(profile, AgentProfile)
        assert 0 <= confidence <= 1

    def test_create_workflow_function(self):
        """Test create_workflow function."""
        execution = create_workflow("Build mobile application")

        assert isinstance(execution, WorkflowExecution)
        assert len(execution.workflow) > 0
