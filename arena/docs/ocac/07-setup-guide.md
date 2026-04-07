# OCAC Project Setup Guide

> Step-by-step: bootstrap ~/ocac as a standalone project with access to ALL your assets.

---

## Architecture

```
~/ocac/                          <-- NEW top-level project
├── .claude/
│   ├── settings.json            <-- Points to ALL skill dirs
│   ├── skills/                  <-- OCAC-specific skills
│   └── agents/                  <-- Agent definitions
├── CLAUDE.md                    <-- Product instructions + references
├── arena/                       <-- Symlink to ralph-orchestrator/arena/
├── docs/                        <-- Copy of arena/docs/ocac/ (local reference)
├── src/                         <-- OCAC source code
└── package.json

~/ralph-orchestrator/            <-- Agent #1 (Ralph orchestrator)
~/expert-ai-skills/              <-- 233 skills library
~/superpowers/                   <-- Core skills v3.6.2
~/BMAD-METHODV7/                 <-- 50+ design workflows
```

**OCAC is the top-level project. Ralph is one agent inside it.**

---

## Step 1: Create the project

```bash
mkdir ~/ocac && cd ~/ocac && git init
```

## Step 2: Clone skill/workflow repos (if not already cloned)

```bash
cd ~
git clone https://github.com/whd4/expert-ai-skills.git
git clone https://github.com/whd4/superpowers.git
git clone https://github.com/whd4/BMAD-METHODV7.git
```

## Step 3: Create `.claude/settings.json`

```bash
mkdir -p ~/ocac/.claude
```

```json
{
  "skills": {
    "dirs": [
      ".claude/skills",
      "../ralph-orchestrator/.claude/skills",
      "../expert-ai-skills/.claude/skills",
      "../superpowers/.claude/skills",
      "../BMAD-METHODV7/.claude/skills"
    ]
  }
}
```

MCPs are global (follow your Claude user profile) — no config needed.

## Step 4: Symlink the Arena

```bash
ln -s ~/ralph-orchestrator/arena ~/ocac/arena
```

This gives OCAC access to:
- All MCP registry configs
- All agent adapters (Ralph, OpenClaw, Hermes)
- Knowledge seed data
- The OCAC docs themselves

## Step 5: Copy OCAC docs locally

```bash
cp -r ~/ralph-orchestrator/arena/docs/ocac ~/ocac/docs
```

## Step 6: Create CLAUDE.md

Create `~/ocac/CLAUDE.md` with:

```markdown
# OCAC — OpenClaw Agentic Companion

> The world's first personal cognitive twin.

## What This Is
- Personal cognitive twin with Deep Confidence + Monte Carlo intelligence
- 6 core systems: BBT, BMA, CIE, RPA, NCC, TPE
- Two tiers: Personal (Phase 1) and Enterprise (Phase 2)

## Documentation
- Full specs in `docs/` (PRD, intelligence stack, core systems, etc.)
- Arena assets linked at `arena/` (agent adapters, MCP registry, knowledge)
- BMAD method at `../BMAD-METHODV7/` (design workflows only)

## Skills Available
- Ralph orchestrator skills (18) via `../ralph-orchestrator/.claude/skills/`
- Expert AI skills (233) via `../expert-ai-skills/.claude/skills/`
- Superpowers skills (111) via `../superpowers/.claude/skills/`
- BMAD workflows (50+) via `../BMAD-METHODV7/.claude/skills/`

## Privacy Requirements (Non-Negotiable)
- Local-first memory
- Encrypted personal state
- User-controlled data ingestion
- No cross-user training
- No medical/mental health claims

## Build & Test
TBD — depends on chosen tech stack (TypeScript recommended for Arena compatibility)
```

## Step 7: Push to GitHub

```bash
cd ~/ocac
git add . && git commit -m "feat: OCAC project bootstrap with full asset access"
# Then create repo on GitHub and push
```

## Step 8: Open Claude Code in ~/ocac

```bash
cd ~/ocac
claude
```

All MCPs, all skills, all agents, all workflows — available from day one.

---

## What You Get

| Asset | Count | Source |
|-------|-------|--------|
| MCP Servers | 13 | Global Claude profile |
| Ralph skills | 18 | ../ralph-orchestrator/.claude/skills/ |
| Expert AI skills | 233 | ../expert-ai-skills/.claude/skills/ |
| Superpowers skills | 111 | ../superpowers/.claude/skills/ |
| BMAD workflows | 50+ | ../BMAD-METHODV7/.claude/skills/ |
| Agent adapters | 3 | arena/adapters/ (Ralph, OpenClaw, Hermes) |
| Knowledge base | 14 entries | arena/knowledge/seed.jsonl |
| OCAC specs | 7 docs | docs/ |

---

## Next Steps After Setup

1. Open Claude Code in `~/ocac`
2. Say: "Read docs/ and arena/HANDOFF.md — bootstrap the OCAC project"
3. Use `/pdd` to transform the PRD into an implementation plan
4. Begin building Phase 1 (Personal MVP)
