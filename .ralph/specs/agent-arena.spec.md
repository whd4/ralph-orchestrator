# Agent Arena — The Detachable Asset Layer

> **Goal:** Build a persistence layer that owns all assets (MCPs, skills, knowledge, data) independently of any agent, enabling agents to be swapped, combined, and competed without losing anything.

## Status: DRAFT — Awaiting Review

## Problem

Today, assets are entangled with agents:
- MCPs are configured per-agent (Claude Code settings, OpenClaw gateway config)
- Skills are stored inside agent repos (`.claude/skills/`, Hermes skill files)
- Knowledge/memories are agent-specific (`.ralph/agent/memories.md`, Hermes memory DB)
- Data connections (Notion, Gmail, Supabase, Figma) are wired to specific agent harnesses

**When you switch agents, you lose your stuff.** When an agent dies (project abandoned, company pivots, API deprecated), your accumulated knowledge dies with it.

## Solution: Agent Arena

A **detachable layer** that sits between agents and assets:

```
┌─────────────────────────────────────────────────────┐
│                    AGENT ARENA                       │
│         (owns everything, agents own nothing)        │
│                                                      │
│  ┌─────────────┐  ┌──────────┐  ┌───────────────┐  │
│  │ Asset Store  │  │ MCP Hub  │  │ Knowledge DB  │  │
│  │ Skills,tools │  │ All MCPs │  │ Memories,docs │  │
│  │ workflows    │  │ unified  │  │ patterns      │  │
│  └──────┬───────┘  └─────┬────┘  └───────┬───────┘  │
│         │                │               │           │
│  ┌──────┴────────────────┴───────────────┴───────┐  │
│  │              AGENT ADAPTER PROTOCOL             │  │
│  │    (standard interface agents plug into)        │  │
│  └──────┬──────────┬──────────┬──────────┬───────┘  │
│         │          │          │          │           │
│    ┌────┴───┐ ┌────┴───┐ ┌───┴────┐ ┌───┴────┐     │
│    │ Ralph  │ │OpenClaw│ │ Hermes │ │ Agent  │     │
│    │  Agent │ │  Agent │ │  Agent │ │   N    │     │
│    └────────┘ └────────┘ └────────┘ └────────┘     │
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │              ARENA ENGINE                     │   │
│  │   Match orchestration, Elo ratings, replay    │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

## Core Principles

1. **Assets are permanent, agents are temporary** — Skills, MCPs, knowledge, and data belong to the Arena, not any individual agent.
2. **Standard adapter protocol** — Every agent (Ralph, OpenClaw, Hermes, custom) plugs in via the same interface.
3. **Hot-swap agents** — Switch from Ralph to Hermes mid-task without losing context.
4. **Compete to discover** — Run the same task against multiple agents simultaneously to find the best agent or combination for the job.
5. **Accumulate, never lose** — When an agent learns something, it writes back to the Arena's Knowledge DB. All agents benefit.

## Architecture

### Layer 1: Asset Store

Owns all portable assets in agent-agnostic formats:

| Asset Type | Storage Format | Example |
|-----------|---------------|---------|
| Skills | Markdown + YAML frontmatter | `skills/code-review.skill.md` |
| Tools | MCP server definitions | `tools/github.mcp.json` |
| Workflows | YAML DAGs | `workflows/pr-review.workflow.yml` |
| Templates | Mustache/Handlebars | `templates/commit-message.hbs` |
| Configs | YAML | `configs/default-agent.yml` |

```
arena/
  assets/
    skills/           # Portable skill definitions
    tools/            # MCP server configs (unified registry)
    workflows/        # Multi-step automation definitions
    templates/        # Prompt templates, output formats
    configs/          # Agent configurations
```

### Layer 2: MCP Hub

A **unified MCP registry** that any agent can access:

```yaml
# arena/tools/registry.yml
mcps:
  github:
    server: "@anthropic/github-mcp"
    scope: ["whd4/ralph-orchestrator", "whd4/BMAD-METHODV7"]
    credentials: vault://github-token
  notion:
    server: "@notionhq/notion-mcp"
    credentials: vault://notion-token
  gmail:
    server: "@anthropic/gmail-mcp"
    credentials: vault://gmail-oauth
  supabase:
    server: "@supabase/mcp"
    credentials: vault://supabase-key
  figma:
    server: "@anthropic/figma-mcp"
    credentials: vault://figma-token
  # ... all 13 of your current MCPs
```

When an agent starts, it receives a filtered view of available MCPs based on the task scope. The Arena handles auth, not the agent.

### Layer 3: Knowledge DB

Unified memory that all agents read from and write to:

```
arena/
  knowledge/
    memories.jsonl      # Accumulated learnings (append-only)
    patterns.jsonl      # Recognized patterns across agents
    decisions.jsonl     # Past decisions and outcomes
    context/            # Per-project context files
      ralph-orchestrator/
      BMAD-METHODV7/
```

Key design: **append-only, never delete.** When Agent A learns something, Agent B can use it next time. Knowledge has provenance (which agent, when, what task).

### Layer 4: Agent Adapter Protocol (AAP)

The standard interface every agent must implement:

```typescript
interface AgentAdapter {
  // Identity
  name: string;                    // "ralph", "openclaw", "hermes"
  version: string;
  capabilities: Capability[];      // ["code", "chat", "browse", "voice"]

  // Lifecycle
  initialize(config: ArenaConfig): Promise<void>;
  shutdown(): Promise<void>;

  // Execution
  execute(task: Task, context: Context): AsyncIterable<Event>;

  // Knowledge
  getLearnedSkills(): Skill[];     // What this agent has learned
  ingestKnowledge(k: Knowledge[]): void;  // Feed arena knowledge in
}

interface Task {
  id: string;
  prompt: string;
  assets: AssetRef[];              // Skills, tools, templates to use
  mcps: McpRef[];                  // Which MCPs are available
  constraints: Constraint[];       // Time limit, token budget, quality gates
}

interface Context {
  knowledge: Knowledge[];          // Relevant memories from Knowledge DB
  history: Event[];                // Previous events in this session
  workspace: string;               // Filesystem path
}
```

### Layer 5: Arena Engine

Orchestrates agent competition and collaboration:

```yaml
# arena/matches/code-review-battle.match.yml
match:
  type: head-to-head
  task: "Review PR #217 for security issues"
  agents: [ralph, hermes]
  scoring:
    - metric: issues_found
      weight: 0.4
    - metric: false_positive_rate
      weight: 0.3
    - metric: time_to_complete
      weight: 0.2
    - metric: human_preference
      weight: 0.1
  sandbox: docker
  timeout: 300s
```

Match types:
- **Head-to-head**: Same task, two agents, blind comparison
- **Collaboration**: Agents work together (e.g., Ralph codes, Hermes reviews)
- **Tournament**: Round-robin across N agents on M tasks
- **Discovery**: Try all agent combinations to find best pairings

## What You Already Have (mapped to Arena layers)

| Arena Layer | Your Current Asset | Location |
|-------------|-------------------|----------|
| Asset Store | 18 Claude Code skills | `.claude/skills/` |
| Asset Store | 233 expert-ai-skills | `whd4/expert-ai-skills` repo |
| Asset Store | superpowers v3.6.2 | `whd4/superpowers` repo |
| MCP Hub | 13 connected MCPs | Claude Code config |
| Knowledge DB | Ralph memories | `.ralph/agent/memories.md` |
| Knowledge DB | BMAD Method v7 | `whd4/BMAD-METHODV7` repo |
| Agent Adapter | Ralph orchestrator | This repo (Rust) |
| Agent Adapter | OpenClaw (to add) | openclaw/openclaw (TS) |
| Agent Adapter | Hermes Agent (to add) | NousResearch/hermes-agent |
| Arena Engine | (to build) | New component |

## Implementation Plan

### Phase 1: Asset Extraction (this session → next session)
Extract assets from agent-specific locations into Arena format:
- [ ] Create `arena/` directory structure
- [ ] Extract skills from `.claude/skills/` into portable format
- [ ] Create unified MCP registry from current Claude Code config
- [ ] Extract memories into arena knowledge DB format

### Phase 2: Agent Adapter Protocol (next session)
Define and implement the AAP:
- [ ] Write AAP TypeScript interface definition
- [ ] Build Ralph adapter (wraps existing Rust CLI)
- [ ] Build OpenClaw adapter (wraps openclaw gateway)
- [ ] Build Hermes adapter (wraps hermes-agent CLI)

### Phase 3: Arena Engine (session after)
Build the competition/collaboration engine:
- [ ] Match orchestrator (run same task against multiple agents)
- [ ] Scoring system (Elo ratings, task metrics)
- [ ] Replay system (full event traces)
- [ ] Leaderboard dashboard

### Phase 4: Knowledge Sharing (ongoing)
Build the learning loop:
- [ ] Agent-to-arena knowledge writeback
- [ ] Cross-agent knowledge injection
- [ ] Pattern recognition across agent outcomes
- [ ] Automatic skill generation from winning strategies

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Arena core | TypeScript/Node.js | Fast prototyping, MCP ecosystem is TS-native |
| Asset Store | Filesystem (Markdown, YAML, JSON) | Git-friendly, human-readable, agent-agnostic |
| MCP Hub | MCP protocol (stdio/SSE) | Standard protocol, all agents speak it |
| Knowledge DB | SQLite + JSONL | Simple, portable, works everywhere |
| Arena Engine | TypeScript + Docker | Sandboxed agent execution |
| Dashboard | React + Vite | Reuse ralph-orchestrator's web stack |
| Leaderboard | SQLite + tRPC | Reuse ralph-orchestrator's backend stack |

## Key Design Decisions

1. **Filesystem-first**: Assets stored as files, not in databases. Git tracks changes. Human-readable always.
2. **MCP as the universal tool protocol**: Every agent already speaks MCP (or can via adapter). Don't invent a new tool protocol.
3. **JSONL for knowledge**: Append-only, streamable, grep-able. Same format Ralph already uses.
4. **Docker for sandboxing**: Agents run in containers for competition. Local execution for development.
5. **No vendor lock-in**: The Arena works with any agent that implements AAP. OpenClaw dies? Unplug it. New agent appears? Plug it in.

## Success Criteria

- [ ] Can swap Ralph for Hermes on a task without reconfiguring MCPs
- [ ] Can run the same task against 3 agents and see who wins
- [ ] Knowledge learned by one agent is available to all others
- [ ] Adding a new agent requires only implementing the AgentAdapter interface
- [ ] Removing an agent loses zero assets, zero knowledge, zero configuration

## Open Questions

1. **Credential management**: Vault? Environment variables? Per-agent scoped tokens?
2. **Knowledge conflict resolution**: When two agents learn contradictory things, which wins?
3. **Real-time vs batch competition**: Live head-to-head or async tournament?
4. **Monetization**: Is this a product or a personal tool?
