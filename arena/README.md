# Agent Arena

> Assets are permanent, agents are temporary.

Agent Arena is the detachable persistence layer between AI agents and your assets.
Swap, combine, and compete agents without losing your MCPs, skills, knowledge, or data.

## Structure

```
arena/
  assets/
    skills/         # Portable skill definitions (agent-agnostic)
    tools/          # Unified MCP registry
    workflows/      # Multi-step automation definitions
    templates/      # Prompt templates, output formats
    configs/        # Agent configurations
  knowledge/        # Append-only knowledge DB
    context/        # Per-project context
  adapters/         # Agent Adapter Protocol + implementations
    protocol.ts     # The AAP interface definition
    ralph-adapter.ts
  engine/           # Match orchestration, scoring, leaderboard
  dashboard/        # React web UI
```

## Quick Start

```bash
# Register an agent
arena register ralph --adapter ./adapters/ralph-adapter.ts

# Run a solo task
arena run --agent ralph --task "Fix the bug in parser.rs"

# Head-to-head competition
arena match --agents ralph,hermes --task "Review PR #42" --scoring balanced

# View leaderboard
arena leaderboard
```

## Core Principle

When you add an agent, it gets access to ALL your assets (MCPs, skills, knowledge).
When you remove an agent, ZERO assets are lost. The Arena owns everything.

## Spec

See [.ralph/specs/agent-arena.spec.md](../.ralph/specs/agent-arena.spec.md) for the full design.
