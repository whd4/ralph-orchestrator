/**
 * Ralph Adapter — Wraps ralph-orchestrator as an Arena agent.
 *
 * Ralph is a Rust-based AI coding orchestrator with:
 * - Hat system (role-based agent selection)
 * - Event loop with backpressure
 * - Parallel worktree execution
 * - Memory and task persistence
 *
 * This adapter translates between Arena's AgentAdapter protocol
 * and Ralph's CLI interface.
 */

import { spawn } from "child_process";
import { readFileSync, existsSync } from "fs";
import { join } from "path";
import type {
  AgentAdapter,
  AgentIdentity,
  AgentEvent,
  Knowledge,
  McpRef,
  Task,
} from "./protocol";

export class RalphAdapter implements AgentAdapter {
  identity: AgentIdentity = {
    name: "ralph",
    version: "2.6.0",
    capabilities: ["code", "plan", "review", "terminal", "multi-agent"],
    maxContextTokens: 176_000,
    costPerMToken: 15, // Claude Opus pricing
  };

  private workDir: string = "";
  private knowledge: Knowledge[] = [];

  async initialize(config: {
    workDir: string;
    mcpHub: McpRef[];
    knowledge: Knowledge[];
  }): Promise<void> {
    this.workDir = config.workDir;
    this.knowledge = config.knowledge;
  }

  async *execute(
    task: Task,
    context: { knowledge: Knowledge[]; history: AgentEvent[] }
  ): AsyncIterable<AgentEvent> {
    const timestamp = () => new Date().toISOString();

    yield {
      type: "started",
      timestamp: timestamp(),
      agent: "ralph",
      data: { taskId: task.id, prompt: task.prompt },
    };

    // Build ralph CLI command
    const args = [
      "run",
      "-p",
      task.prompt,
      "--max-iterations",
      "5",
      "--headless",
    ];

    // Add workspace if specified
    if (task.workspace) {
      args.push("--workdir", task.workspace);
    }

    const proc = spawn("ralph", args, {
      cwd: this.workDir,
      env: {
        ...process.env,
        RALPH_DIAGNOSTICS: "1", // Capture all events for Arena replay
      },
    });

    // Stream stdout as text events
    for await (const chunk of proc.stdout) {
      yield {
        type: "text",
        timestamp: timestamp(),
        agent: "ralph",
        data: { text: chunk.toString() },
      };
    }

    // Wait for process to exit
    const exitCode = await new Promise<number>((resolve) => {
      proc.on("close", (code) => resolve(code ?? 1));
    });

    // Read diagnostics if available
    const learned = this.extractKnowledge();

    yield {
      type: "completed",
      timestamp: timestamp(),
      agent: "ralph",
      data: {
        success: exitCode === 0,
        summary: exitCode === 0 ? "Task completed" : "Task failed",
        tokensUsed: 0, // TODO: extract from diagnostics
        knowledgeLearned: learned,
        artifacts: [], // TODO: extract from git diff
      },
    };
  }

  getLearnedKnowledge(): Knowledge[] {
    // Read from Ralph's memory file
    const memoryPath = join(this.workDir, ".ralph", "agent", "memories.md");
    if (!existsSync(memoryPath)) return [];

    const content = readFileSync(memoryPath, "utf-8");
    return [
      {
        id: `ralph-memory-${Date.now()}`,
        source: "ralph",
        timestamp: new Date().toISOString(),
        type: "memory",
        content,
        confidence: 0.8,
        context: { project: "ralph-orchestrator" },
      },
    ];
  }

  ingestKnowledge(knowledge: Knowledge[]): void {
    this.knowledge = [...this.knowledge, ...knowledge];
    // TODO: Write cross-agent knowledge into Ralph's context files
  }

  async shutdown(): Promise<void> {
    // Ralph processes clean up on their own via lock files
  }
}
