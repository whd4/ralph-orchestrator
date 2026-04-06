#!/usr/bin/env npx ts-node
/**
 * Agent Arena CLI — The command-line interface for managing agents,
 * running matches, and viewing the leaderboard.
 *
 * Usage:
 *   arena agents              — List registered agents and their status
 *   arena run <prompt>        — Run a task against a single agent
 *   arena match <prompt>      — Head-to-head competition between agents
 *   arena leaderboard         — View agent rankings
 *   arena assets              — List all available assets
 *   arena knowledge           — Query the Knowledge DB
 */

import { RalphAdapter } from "./adapters/ralph-adapter";
import { OpenClawAdapter } from "./adapters/openclaw-adapter";
import { HermesAdapter } from "./adapters/hermes-adapter";
import type { AgentAdapter, AgentIdentity, MatchResult } from "./adapters/protocol";

// ─── Agent Registry ───────────────────────────────────────────────────

const AGENTS: Record<string, AgentAdapter> = {
  ralph: new RalphAdapter(),
  openclaw: new OpenClawAdapter(),
  hermes: new HermesAdapter(),
};

// ─── Commands ─────────────────────────────────────────────────────────

function listAgents(): void {
  console.log("\n  AGENT ARENA — Registered Agents\n");
  console.log("  %-12s %-10s %-8s %s", "NAME", "VERSION", "COST", "CAPABILITIES");
  console.log("  " + "─".repeat(60));

  for (const [name, adapter] of Object.entries(AGENTS)) {
    const id = adapter.identity;
    console.log(
      "  %-12s %-10s $%-7s %s",
      id.name,
      id.version,
      (id.costPerMToken || "?") + "/Mt",
      id.capabilities.join(", ")
    );
  }
  console.log();
}

async function runTask(agentName: string, prompt: string): Promise<void> {
  const adapter = AGENTS[agentName];
  if (!adapter) {
    console.error(`Unknown agent: ${agentName}. Available: ${Object.keys(AGENTS).join(", ")}`);
    process.exit(1);
  }

  console.log(`\n  Running task with ${agentName}...`);
  console.log(`  Prompt: "${prompt}"\n`);

  await adapter.initialize({
    workDir: process.cwd(),
    mcpHub: [],
    knowledge: [],
  });

  for await (const event of adapter.execute(
    {
      id: `task-${Date.now()}`,
      prompt,
      assets: [],
      mcps: [],
      constraints: [],
      workspace: process.cwd(),
    },
    { knowledge: [], history: [] }
  )) {
    if (event.type === "text") {
      process.stdout.write((event.data as { text: string }).text);
    } else if (event.type === "completed") {
      const data = event.data as { success: boolean; summary: string };
      console.log(`\n\n  Result: ${data.success ? "SUCCESS" : "FAILED"} — ${data.summary}`);
    }
  }

  await adapter.shutdown();
}

async function runMatch(agents: string[], prompt: string): Promise<void> {
  console.log(`\n  HEAD-TO-HEAD: ${agents.join(" vs ")}`);
  console.log(`  Task: "${prompt}"\n`);

  const results: Record<string, { output: string; duration: number; success: boolean }> = {};

  for (const agentName of agents) {
    const adapter = AGENTS[agentName];
    if (!adapter) {
      console.error(`Unknown agent: ${agentName}`);
      continue;
    }

    const start = Date.now();
    let output = "";

    await adapter.initialize({
      workDir: process.cwd(),
      mcpHub: [],
      knowledge: [],
    });

    for await (const event of adapter.execute(
      {
        id: `match-${Date.now()}`,
        prompt,
        assets: [],
        mcps: [],
        constraints: [],
        workspace: process.cwd(),
      },
      { knowledge: [], history: [] }
    )) {
      if (event.type === "text") {
        output += (event.data as { text: string }).text;
      }
    }

    results[agentName] = {
      output: output.slice(0, 500),
      duration: Date.now() - start,
      success: true,
    };

    await adapter.shutdown();
  }

  // Display results
  console.log("\n  RESULTS\n");
  for (const [name, result] of Object.entries(results)) {
    console.log(`  ${name} (${result.duration}ms):`);
    console.log(`    ${result.output.slice(0, 200)}...`);
    console.log();
  }
}

// ─── Main ─────────────────────────────────────────────────────────────

const [,, command, ...args] = process.argv;

switch (command) {
  case "agents":
    listAgents();
    break;
  case "run":
    if (args.length < 2) {
      console.error("Usage: arena run <agent> <prompt>");
      process.exit(1);
    }
    runTask(args[0], args.slice(1).join(" "));
    break;
  case "match":
    if (args.length < 3) {
      console.error("Usage: arena match <agent1>,<agent2> <prompt>");
      process.exit(1);
    }
    runMatch(args[0].split(","), args.slice(1).join(" "));
    break;
  case "assets":
    console.log("\n  See arena/assets/skills/manifest.yml for skill inventory");
    console.log("  See arena/assets/tools/mcp-registry.yml for MCP registry\n");
    break;
  case "knowledge":
    console.log("\n  See arena/knowledge/seed.jsonl for knowledge DB\n");
    break;
  default:
    console.log(`
  Agent Arena — Assets are permanent, agents are temporary.

  Commands:
    agents              List registered agents
    run <agent> <prompt> Run a task with one agent
    match <a1>,<a2> <prompt> Head-to-head competition
    assets              List available assets
    knowledge           Query the Knowledge DB

  Registered agents: ${Object.keys(AGENTS).join(", ")}
`);
}
