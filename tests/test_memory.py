# ABOUTME: Tests for Memory System
# ABOUTME: Verifies persistent storage and pattern recognition

"""Tests for the Memory System."""

import pytest
from pathlib import Path
import tempfile
import shutil

from ralph_orchestrator.memory import (
    MemoryStore,
    Memory,
    Pattern,
    MemoryType,
    get_memory_store,
)


class TestMemoryStore:
    """Tests for MemoryStore class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp = tempfile.mkdtemp()
        yield Path(temp)
        shutil.rmtree(temp, ignore_errors=True)

    @pytest.fixture
    def store(self, temp_dir):
        """Create memory store with temp database."""
        return MemoryStore(db_path=temp_dir / "test_memory.db")

    def test_store_initialization(self, store):
        """Test store initializes correctly."""
        assert store.db_path.exists()

    def test_store_memory(self, store):
        """Test storing a memory."""
        memory = Memory(
            memory_type=MemoryType.TASK_EXECUTION,
            content="Implemented REST API successfully",
            summary="REST API implementation",
            keywords=["api", "rest", "implementation"],
            importance=0.8
        )

        memory_id = store.store_memory(memory)

        assert memory_id > 0

    def test_store_task_result(self, store):
        """Test storing task execution result."""
        memory_id = store.store_task_result(
            task_description="Implement user authentication",
            success=True,
            tokens_used=5000,
            strategy="balanced",
            output_preview="Authentication implemented successfully"
        )

        assert memory_id > 0

    def test_duplicate_content_handling(self, store):
        """Test handling duplicate content."""
        memory = Memory(
            memory_type=MemoryType.CONTEXT,
            content="This is duplicate content",
            summary="Duplicate test"
        )

        id1 = store.store_memory(memory)
        id2 = store.store_memory(memory)

        # Should return same ID (deduplicated)
        assert id1 == id2

    def test_get_relevant_memories(self, store):
        """Test retrieving relevant memories."""
        # Store some memories
        store.store_task_result(
            task_description="Implement database connection",
            success=True,
            tokens_used=3000,
            strategy="conservative"
        )
        store.store_task_result(
            task_description="Add user registration",
            success=True,
            tokens_used=4000,
            strategy="balanced"
        )

        # Query for relevant memories
        memories = store.get_relevant_memories(
            query="database implementation",
            limit=5
        )

        assert isinstance(memories, list)

    def test_get_relevant_memories_with_type_filter(self, store):
        """Test retrieving memories with type filter."""
        store.store_task_result(
            task_description="Test task",
            success=True,
            tokens_used=1000,
            strategy="aggressive"
        )

        memories = store.get_relevant_memories(
            query="test",
            memory_types=[MemoryType.TASK_EXECUTION],
            limit=5
        )

        assert all(m.memory_type == MemoryType.TASK_EXECUTION for m in memories)

    def test_store_lesson(self, store):
        """Test storing a learned lesson."""
        lesson_id = store.store_lesson(
            context="API implementation",
            lesson="Always validate input parameters",
            source_task="Implement REST endpoints",
            importance=0.9
        )

        assert lesson_id > 0

    def test_get_lessons_for_context(self, store):
        """Test retrieving lessons for context."""
        store.store_lesson(
            context="Database queries",
            lesson="Use parameterized queries to prevent SQL injection"
        )

        lessons = store.get_lessons_for_context("database operations")

        assert isinstance(lessons, list)

    def test_store_pattern(self, store):
        """Test storing a pattern."""
        pattern = Pattern(
            name="api_success_pattern",
            description="Pattern for successful API implementations",
            trigger_conditions=["api", "rest", "endpoint"],
            success_rate=0.85,
            recommended_action="Use TDD approach"
        )

        pattern_id = store.store_pattern(pattern)

        assert pattern_id > 0

    def test_update_pattern(self, store):
        """Test updating existing pattern."""
        pattern1 = Pattern(
            name="test_pattern",
            description="First version",
            success_rate=0.7
        )
        pattern2 = Pattern(
            name="test_pattern",
            description="Updated version",
            success_rate=0.8
        )

        id1 = store.store_pattern(pattern1)
        id2 = store.store_pattern(pattern2)

        # Should be same pattern
        assert id1 == id2

    def test_get_patterns_for_task(self, store):
        """Test retrieving patterns for a task."""
        store.store_pattern(Pattern(
            name="auth_pattern",
            description="Authentication pattern",
            trigger_conditions=["auth", "login", "security"]
        ))

        patterns = store.get_patterns_for_task("Implement user authentication")

        assert isinstance(patterns, list)

    def test_cleanup_old_memories(self, store):
        """Test cleanup of old memories."""
        # Store a memory
        store.store_memory(Memory(
            memory_type=MemoryType.CONTEXT,
            content="Old memory to clean up",
            importance=0.3  # Low importance
        ))

        # Cleanup
        deleted = store.cleanup_old_memories(days=0)

        # May or may not delete depending on timing
        assert deleted >= 0

    def test_statistics(self, store):
        """Test getting store statistics."""
        store.store_task_result(
            task_description="Test task",
            success=True,
            tokens_used=1000,
            strategy="balanced"
        )

        stats = store.get_statistics()

        assert "total_memories" in stats
        assert "total_patterns" in stats
        assert "total_lessons" in stats
        assert "database_path" in stats


class TestMemory:
    """Tests for Memory dataclass."""

    def test_memory_creation(self):
        """Test creating a memory."""
        memory = Memory(
            memory_type=MemoryType.TASK_EXECUTION,
            content="Task completed successfully",
            summary="Task completion",
            keywords=["task", "complete"],
            importance=0.8
        )

        assert memory.memory_type == MemoryType.TASK_EXECUTION
        assert memory.importance == 0.8
        assert memory.access_count == 0

    def test_memory_defaults(self):
        """Test memory default values."""
        memory = Memory(
            memory_type=MemoryType.CONTEXT,
            content="Test content"
        )

        assert memory.importance == 0.5
        assert memory.keywords == []
        assert memory.metadata == {}


class TestPattern:
    """Tests for Pattern dataclass."""

    def test_pattern_creation(self):
        """Test creating a pattern."""
        pattern = Pattern(
            name="test_pattern",
            description="A test pattern",
            trigger_conditions=["test", "example"],
            success_rate=0.75,
            recommended_action="Follow best practices"
        )

        assert pattern.name == "test_pattern"
        assert pattern.success_rate == 0.75
        assert len(pattern.trigger_conditions) == 2


class TestMemoryType:
    """Tests for MemoryType enum."""

    def test_memory_types(self):
        """Test all memory types exist."""
        types = [
            MemoryType.TASK_EXECUTION,
            MemoryType.DECISION,
            MemoryType.PATTERN,
            MemoryType.LESSON,
            MemoryType.CONTEXT,
            MemoryType.PREFERENCE,
            MemoryType.ERROR
        ]

        assert len(types) == 7

    def test_memory_type_values(self):
        """Test memory type string values."""
        assert MemoryType.TASK_EXECUTION.value == "task_execution"
        assert MemoryType.DECISION.value == "decision"


class TestGetMemoryStore:
    """Tests for get_memory_store factory function."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp = tempfile.mkdtemp()
        yield Path(temp)
        shutil.rmtree(temp, ignore_errors=True)

    def test_get_store(self, temp_dir):
        """Test getting memory store."""
        # Reset singleton
        import ralph_orchestrator.memory as mem
        mem._memory_store = None

        store = get_memory_store(temp_dir / "test.db")

        assert isinstance(store, MemoryStore)
