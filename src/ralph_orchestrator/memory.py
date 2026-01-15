# ABOUTME: Enhanced memory system for learning from interactions
# ABOUTME: SQLite-based persistent storage with pattern recognition

"""Enhanced Memory System for Ralph Orchestrator.

This module provides a persistent memory system that learns from
interactions, stores patterns, and improves decision-making over time.
"""

import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import re
from collections import Counter

logger = logging.getLogger(__name__)


class MemoryType(str, Enum):
    """Types of memories stored."""
    TASK_EXECUTION = "task_execution"       # Task execution results
    DECISION = "decision"                    # Strategic decisions
    PATTERN = "pattern"                      # Recognized patterns
    LESSON = "lesson"                        # Learned lessons
    CONTEXT = "context"                      # Contextual information
    PREFERENCE = "preference"                # User preferences
    ERROR = "error"                          # Error patterns


@dataclass
class Memory:
    """A single memory entry."""
    id: Optional[int] = None
    memory_type: MemoryType = MemoryType.CONTEXT
    content: str = ""
    summary: str = ""
    keywords: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5  # 0-1 scale
    access_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None


@dataclass
class Pattern:
    """A recognized pattern from multiple interactions."""
    id: Optional[int] = None
    name: str = ""
    description: str = ""
    trigger_conditions: List[str] = field(default_factory=list)
    success_rate: float = 0.5
    occurrence_count: int = 0
    recommended_action: str = ""
    examples: List[Dict] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)


class MemoryStore:
    """SQLite-based persistent memory store.

    Provides:
    - Persistent storage of memories and patterns
    - Keyword-based search and retrieval
    - Pattern recognition and learning
    - Automatic cleanup of old memories
    - Relevance scoring for retrieval
    """

    SCHEMA_VERSION = 1

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize the memory store.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path or Path(".agent/memory.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_database()
        logger.info(f"Memory store initialized at {self.db_path}")

    def _init_database(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    summary TEXT,
                    keywords TEXT,
                    metadata TEXT,
                    importance REAL DEFAULT 0.5,
                    access_count INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TEXT DEFAULT CURRENT_TIMESTAMP,
                    expires_at TEXT,
                    content_hash TEXT UNIQUE
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE,
                    description TEXT,
                    trigger_conditions TEXT,
                    success_rate REAL DEFAULT 0.5,
                    occurrence_count INTEGER DEFAULT 0,
                    recommended_action TEXT,
                    examples TEXT,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    last_updated TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS lessons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    context TEXT NOT NULL,
                    lesson TEXT NOT NULL,
                    source_task TEXT,
                    importance REAL DEFAULT 0.5,
                    times_applied INTEGER DEFAULT 0,
                    success_rate REAL DEFAULT 0.5,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    last_applied TEXT
                )
            """)

            # Create indexes for faster searching
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_type
                ON memories(memory_type)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_keywords
                ON memories(keywords)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_patterns_name
                ON patterns(name)
            """)

            # Metadata table for schema versioning
            conn.execute("""
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)

            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                ("schema_version", str(self.SCHEMA_VERSION))
            )

            conn.commit()

    def _content_hash(self, content: str) -> str:
        """Generate hash for content deduplication."""
        return hashlib.md5(content.encode()).hexdigest()

    def store_memory(self, memory: Memory) -> int:
        """Store a memory in the database.

        Args:
            memory: Memory to store

        Returns:
            Memory ID
        """
        content_hash = self._content_hash(memory.content)

        with sqlite3.connect(self.db_path) as conn:
            try:
                cursor = conn.execute("""
                    INSERT INTO memories
                    (memory_type, content, summary, keywords, metadata,
                     importance, access_count, created_at, last_accessed,
                     expires_at, content_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    memory.memory_type.value,
                    memory.content,
                    memory.summary,
                    json.dumps(memory.keywords),
                    json.dumps(memory.metadata),
                    memory.importance,
                    memory.access_count,
                    memory.created_at.isoformat(),
                    memory.last_accessed.isoformat(),
                    memory.expires_at.isoformat() if memory.expires_at else None,
                    content_hash
                ))
                conn.commit()
                memory_id = cursor.lastrowid
                logger.debug(f"Stored memory {memory_id}: {memory.summary[:50]}")
                return memory_id
            except sqlite3.IntegrityError:
                # Duplicate content, update access count instead
                conn.execute("""
                    UPDATE memories
                    SET access_count = access_count + 1,
                        last_accessed = ?
                    WHERE content_hash = ?
                """, (datetime.now().isoformat(), content_hash))
                conn.commit()

                # Get existing ID
                result = conn.execute(
                    "SELECT id FROM memories WHERE content_hash = ?",
                    (content_hash,)
                ).fetchone()
                return result[0] if result else -1

    def store_task_result(
        self,
        task_description: str,
        success: bool,
        tokens_used: int,
        strategy: str,
        output_preview: str = "",
        error: str = "",
        metadata: Optional[Dict] = None
    ) -> int:
        """Store task execution result.

        Args:
            task_description: What the task was
            success: Whether it succeeded
            tokens_used: Tokens consumed
            strategy: Strategy used
            output_preview: Preview of output
            error: Error message if failed
            metadata: Additional metadata

        Returns:
            Memory ID
        """
        keywords = self._extract_keywords(task_description)
        keywords.append(strategy)
        keywords.append("success" if success else "failure")

        memory = Memory(
            memory_type=MemoryType.TASK_EXECUTION,
            content=task_description,
            summary=f"{'Success' if success else 'Failure'}: {task_description[:100]}",
            keywords=keywords,
            importance=0.7 if success else 0.6,
            metadata={
                "success": success,
                "tokens_used": tokens_used,
                "strategy": strategy,
                "output_preview": output_preview[:500] if output_preview else "",
                "error": error,
                **(metadata or {})
            }
        )

        memory_id = self.store_memory(memory)

        # Update patterns based on result
        self._update_patterns_from_result(task_description, strategy, success)

        return memory_id

    def store_lesson(
        self,
        context: str,
        lesson: str,
        source_task: str = "",
        importance: float = 0.5
    ) -> int:
        """Store a learned lesson.

        Args:
            context: Context where lesson applies
            lesson: The lesson learned
            source_task: Task that led to lesson
            importance: How important the lesson is

        Returns:
            Lesson ID
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO lessons (context, lesson, source_task, importance)
                VALUES (?, ?, ?, ?)
            """, (context, lesson, source_task, importance))
            conn.commit()
            return cursor.lastrowid

    def get_relevant_memories(
        self,
        query: str,
        memory_types: Optional[List[MemoryType]] = None,
        limit: int = 10,
        min_importance: float = 0.0
    ) -> List[Memory]:
        """Retrieve relevant memories for a query.

        Args:
            query: Search query
            memory_types: Filter by types
            limit: Maximum results
            min_importance: Minimum importance threshold

        Returns:
            List of relevant memories
        """
        keywords = self._extract_keywords(query)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Build query
            sql = """
                SELECT *, 0 as score
                FROM memories
                WHERE importance >= ?
            """
            params: List[Any] = [min_importance]

            if memory_types:
                placeholders = ','.join('?' * len(memory_types))
                sql += f" AND memory_type IN ({placeholders})"
                params.extend([mt.value for mt in memory_types])

            sql += " ORDER BY importance DESC, last_accessed DESC LIMIT ?"
            params.append(limit * 2)  # Get more than needed for scoring

            rows = conn.execute(sql, params).fetchall()

            # Score and filter results
            scored_memories = []
            for row in rows:
                memory_keywords = json.loads(row['keywords'] or '[]')
                # Score based on keyword overlap
                overlap = len(set(keywords) & set(memory_keywords))
                score = overlap / max(len(keywords), 1)

                memory = self._row_to_memory(row)
                scored_memories.append((score, memory))

            # Sort by score and return top results
            scored_memories.sort(key=lambda x: x[0], reverse=True)
            return [m for _, m in scored_memories[:limit]]

    def get_lessons_for_context(
        self,
        context: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get relevant lessons for a context.

        Args:
            context: Context description
            limit: Maximum lessons to return

        Returns:
            List of lesson dictionaries
        """
        keywords = self._extract_keywords(context)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            lessons = []
            for keyword in keywords:
                rows = conn.execute("""
                    SELECT * FROM lessons
                    WHERE context LIKE ?
                    ORDER BY importance DESC, times_applied DESC
                    LIMIT ?
                """, (f"%{keyword}%", limit)).fetchall()

                for row in rows:
                    lessons.append({
                        "id": row["id"],
                        "context": row["context"],
                        "lesson": row["lesson"],
                        "importance": row["importance"],
                        "times_applied": row["times_applied"],
                        "success_rate": row["success_rate"]
                    })

            # Deduplicate and return
            seen_ids = set()
            unique_lessons = []
            for lesson in lessons:
                if lesson["id"] not in seen_ids:
                    seen_ids.add(lesson["id"])
                    unique_lessons.append(lesson)

            return unique_lessons[:limit]

    def _row_to_memory(self, row: sqlite3.Row) -> Memory:
        """Convert database row to Memory object."""
        return Memory(
            id=row['id'],
            memory_type=MemoryType(row['memory_type']),
            content=row['content'],
            summary=row['summary'] or "",
            keywords=json.loads(row['keywords'] or '[]'),
            metadata=json.loads(row['metadata'] or '{}'),
            importance=row['importance'],
            access_count=row['access_count'],
            created_at=datetime.fromisoformat(row['created_at']),
            last_accessed=datetime.fromisoformat(row['last_accessed']),
            expires_at=datetime.fromisoformat(row['expires_at']) if row['expires_at'] else None
        )

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text.

        Args:
            text: Input text

        Returns:
            List of keywords
        """
        # Simple keyword extraction
        # Remove common words and punctuation
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
            'for', 'on', 'with', 'at', 'by', 'from', 'up', 'about', 'into',
            'over', 'after', 'and', 'or', 'but', 'if', 'then', 'else',
            'when', 'than', 'so', 'as', 'that', 'this', 'it', 'its'
        }

        # Tokenize and clean
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', text.lower())

        # Filter and return
        keywords = [
            w for w in words
            if w not in stopwords and len(w) > 2
        ]

        # Get most common
        counter = Counter(keywords)
        return [word for word, _ in counter.most_common(10)]

    def store_pattern(self, pattern: Pattern) -> int:
        """Store or update a pattern.

        Args:
            pattern: Pattern to store

        Returns:
            Pattern ID
        """
        with sqlite3.connect(self.db_path) as conn:
            try:
                cursor = conn.execute("""
                    INSERT INTO patterns
                    (name, description, trigger_conditions, success_rate,
                     occurrence_count, recommended_action, examples, metadata,
                     created_at, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pattern.name,
                    pattern.description,
                    json.dumps(pattern.trigger_conditions),
                    pattern.success_rate,
                    pattern.occurrence_count,
                    pattern.recommended_action,
                    json.dumps(pattern.examples),
                    json.dumps(pattern.metadata),
                    pattern.created_at.isoformat(),
                    pattern.last_updated.isoformat()
                ))
                conn.commit()
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                # Pattern exists, update it
                conn.execute("""
                    UPDATE patterns
                    SET description = ?,
                        trigger_conditions = ?,
                        success_rate = ?,
                        occurrence_count = occurrence_count + 1,
                        recommended_action = ?,
                        examples = ?,
                        metadata = ?,
                        last_updated = ?
                    WHERE name = ?
                """, (
                    pattern.description,
                    json.dumps(pattern.trigger_conditions),
                    pattern.success_rate,
                    pattern.recommended_action,
                    json.dumps(pattern.examples),
                    json.dumps(pattern.metadata),
                    datetime.now().isoformat(),
                    pattern.name
                ))
                conn.commit()

                result = conn.execute(
                    "SELECT id FROM patterns WHERE name = ?",
                    (pattern.name,)
                ).fetchone()
                return result[0] if result else -1

    def get_patterns_for_task(
        self,
        task_description: str,
        limit: int = 5
    ) -> List[Pattern]:
        """Get relevant patterns for a task.

        Args:
            task_description: Task description
            limit: Maximum patterns to return

        Returns:
            List of relevant patterns
        """
        keywords = self._extract_keywords(task_description)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            patterns = []
            for keyword in keywords:
                rows = conn.execute("""
                    SELECT * FROM patterns
                    WHERE trigger_conditions LIKE ?
                       OR description LIKE ?
                    ORDER BY success_rate DESC, occurrence_count DESC
                    LIMIT ?
                """, (f"%{keyword}%", f"%{keyword}%", limit)).fetchall()

                for row in rows:
                    pattern = Pattern(
                        id=row['id'],
                        name=row['name'],
                        description=row['description'],
                        trigger_conditions=json.loads(row['trigger_conditions'] or '[]'),
                        success_rate=row['success_rate'],
                        occurrence_count=row['occurrence_count'],
                        recommended_action=row['recommended_action'],
                        examples=json.loads(row['examples'] or '[]'),
                        metadata=json.loads(row['metadata'] or '{}'),
                        created_at=datetime.fromisoformat(row['created_at']),
                        last_updated=datetime.fromisoformat(row['last_updated'])
                    )
                    patterns.append(pattern)

            # Deduplicate by name
            seen = set()
            unique_patterns = []
            for p in patterns:
                if p.name not in seen:
                    seen.add(p.name)
                    unique_patterns.append(p)

            return unique_patterns[:limit]

    def _update_patterns_from_result(
        self,
        task_description: str,
        strategy: str,
        success: bool
    ) -> None:
        """Update patterns based on task result.

        Args:
            task_description: Task description
            strategy: Strategy used
            success: Whether it succeeded
        """
        keywords = self._extract_keywords(task_description)

        # Create a pattern name from keywords and strategy
        pattern_name = f"{strategy}_{keywords[0] if keywords else 'general'}"

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Check if pattern exists
            row = conn.execute(
                "SELECT * FROM patterns WHERE name = ?",
                (pattern_name,)
            ).fetchone()

            if row:
                # Update existing pattern
                old_success_rate = row['success_rate']
                occurrence_count = row['occurrence_count']

                # Running average
                new_success_rate = (
                    (old_success_rate * occurrence_count + (1 if success else 0)) /
                    (occurrence_count + 1)
                )

                conn.execute("""
                    UPDATE patterns
                    SET success_rate = ?,
                        occurrence_count = occurrence_count + 1,
                        last_updated = ?
                    WHERE name = ?
                """, (new_success_rate, datetime.now().isoformat(), pattern_name))
            else:
                # Create new pattern
                conn.execute("""
                    INSERT INTO patterns
                    (name, description, trigger_conditions, success_rate,
                     occurrence_count, recommended_action, examples, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pattern_name,
                    f"Pattern for {strategy} strategy on {keywords[0] if keywords else 'general'} tasks",
                    json.dumps(keywords),
                    1.0 if success else 0.0,
                    1,
                    f"Consider {strategy} approach",
                    json.dumps([{"task": task_description[:100], "success": success}]),
                    json.dumps({})
                ))

            conn.commit()

    def cleanup_old_memories(
        self,
        days: int = 30,
        keep_important: bool = True
    ) -> int:
        """Clean up old memories.

        Args:
            days: Delete memories older than this
            keep_important: Keep high-importance memories

        Returns:
            Number of deleted memories
        """
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            if keep_important:
                cursor = conn.execute("""
                    DELETE FROM memories
                    WHERE created_at < ?
                      AND importance < 0.7
                      AND expires_at IS NOT NULL
                      AND expires_at < ?
                """, (cutoff, datetime.now().isoformat()))
            else:
                cursor = conn.execute("""
                    DELETE FROM memories
                    WHERE created_at < ?
                """, (cutoff,))

            conn.commit()
            deleted = cursor.rowcount
            logger.info(f"Cleaned up {deleted} old memories")
            return deleted

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory store statistics.

        Returns:
            Statistics dictionary
        """
        with sqlite3.connect(self.db_path) as conn:
            memories_count = conn.execute(
                "SELECT COUNT(*) FROM memories"
            ).fetchone()[0]

            patterns_count = conn.execute(
                "SELECT COUNT(*) FROM patterns"
            ).fetchone()[0]

            lessons_count = conn.execute(
                "SELECT COUNT(*) FROM lessons"
            ).fetchone()[0]

            type_counts = {}
            for row in conn.execute("""
                SELECT memory_type, COUNT(*) as count
                FROM memories
                GROUP BY memory_type
            """).fetchall():
                type_counts[row[0]] = row[1]

            avg_importance = conn.execute(
                "SELECT AVG(importance) FROM memories"
            ).fetchone()[0] or 0.0

            avg_pattern_success = conn.execute(
                "SELECT AVG(success_rate) FROM patterns"
            ).fetchone()[0] or 0.0

            return {
                "total_memories": memories_count,
                "total_patterns": patterns_count,
                "total_lessons": lessons_count,
                "memories_by_type": type_counts,
                "average_importance": avg_importance,
                "average_pattern_success_rate": avg_pattern_success,
                "database_path": str(self.db_path),
                "database_size_kb": self.db_path.stat().st_size / 1024
            }


# Singleton instance
_memory_store: Optional[MemoryStore] = None


def get_memory_store(db_path: Optional[Path] = None) -> MemoryStore:
    """Get or create the memory store singleton.

    Args:
        db_path: Optional custom database path

    Returns:
        Memory store instance
    """
    global _memory_store

    if _memory_store is None:
        _memory_store = MemoryStore(db_path)

    return _memory_store
