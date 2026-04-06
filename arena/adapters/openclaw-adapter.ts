/**
 * OpenClaw Adapter — Wraps OpenClaw as an Arena agent.
 *
 * OpenClaw is a TypeScript/Node.js AI agent gateway with:
 * - 24+ messaging channel integrations
 * - WebSocket control plane (gateway)
 * - Agent isolation via workspaces
 * - ACP (Agent Control Protocol) support
 * - Skills system and plugin architecture
 *
 * Installed at: /opt/node22/bin/openclaw
 * Version: 2026.4.5 (3e72c03)
 *
 * This adapter communicates with OpenClaw via its CLI and gateway WebSocket.
 */

import { spawn } from "child_process";
import type {
  AgentAdapter,
  AgentIdentity,
  AgentEvent,
  Knowledge,
  McpRef,
  Task,
} from "./protocol";

export class OpenClawAdapter implements AgentAdapter {
  identity: AgentIdentity = {
    name: "openclaw",
    version: "2026.4.5",
    capabilities: ["chat", "browse", "voice", "terminal", "multi-agent"],
    maxContextTokens: 200_000, // Depends on underlying model
    costPerMToken: 3, // Variable — depends on model routing
  };

  private workDir: string = "";
  private knowledge: Knowledge[] = [];
  private gatewayPort: number = 18789;

  async initialize(config: {
    workDir: string;
    mcpHub: McpRef[];
    knowledge: Knowledge[];
  }): Promise<void> {
    this.workDir = config.workDir;
    this.knowledge = config.knowledge;

    // OpenClaw uses profiles for isolation — create an arena profile
    const setup = spawn("openclaw", [
      "--profile", "arena",
      "setup",
      "--non-interactive",
    ]);
    await new Promise<void>((resolve) => setup.on("close", () => resolve()));
  }

  async *execute(
    task: Task,
    context: { knowledge: Knowledge[]; history: AgentEvent[] }
  ): AsyncIterable<AgentEvent> {
    const timestamp = () => new Date().toISOString();

    yield {
      type: "started",
      timestamp: timestamp(),
      agent: "openclaw",
      data: { taskId: task.id, prompt: task.prompt },
    };

    // Use OpenClaw's agent command for single-turn execution
    const args = [
      "--profile", "arena",
      "agent",
      "--message", task.prompt,
      "--json", // Structured output
    ];

    const proc = spawn("openclaw", args, {
      cwd: this.workDir,
    });

    let output = "";
    for await (const chunk of proc.stdout) {
      const text = chunk.toString();
      output += text;
      yield {
        type: "text",
        timestamp: timestamp(),
        agent: "openclaw",
        data: { text },
      };
    }

    const exitCode = await new Promise<number>((resolve) => {
      proc.on("close", (code) => resolve(code ?? 1));
    });

    yield {
      type: "completed",
      timestamp: timestamp(),
      agent: "openclaw",
      data: {
        success: exitCode === 0,
        summary: exitCode === 0 ? "Task completed" : "Task failed",
        tokensUsed: 0,
        knowledgeLearned: [],
        artifacts: [],
      },
    };
  }

  getLearnedKnowledge(): Knowledge[] {
    // OpenClaw stores sessions — could extract from ~/.openclaw-arena/sessions/
    return [];
  }

  ingestKnowledge(knowledge: Knowledge[]): void {
    this.knowledge = [...this.knowledge, ...knowledge];
  }

  async shutdown(): Promise<void> {
    // Kill the gateway if running
    spawn("openclaw", ["--profile", "arena", "gateway", "--stop"]);
  }
}
