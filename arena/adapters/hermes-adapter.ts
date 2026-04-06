/**
 * Hermes Agent Adapter — Wraps Hermes Agent as an Arena agent.
 *
 * Hermes Agent is a Python-based self-improving AI runtime with:
 * - Closed learning loop (creates skills from experience)
 * - 200+ model support (Nous Portal, OpenRouter, Ollama)
 * - 40+ built-in tools
 * - 6 terminal backends (local, Docker, SSH, Daytona, Singularity, Modal)
 * - Profiles for isolated instances
 *
 * Installed at: /tmp/hermes-agent (editable pip install)
 * Version: 0.7.0
 * CLI: python3 /tmp/hermes-agent/cli.py
 *
 * This adapter communicates with Hermes via its CLI in query mode.
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

export class HermesAdapter implements AgentAdapter {
  identity: AgentIdentity = {
    name: "hermes",
    version: "0.7.0",
    capabilities: ["code", "chat", "browse", "research", "terminal", "multi-agent"],
    maxContextTokens: 128_000, // Depends on underlying model
    costPerMToken: 2, // Variable — cheapest with Nous Portal
  };

  private workDir: string = "";
  private knowledge: Knowledge[] = [];
  private hermesCliPath: string = "/tmp/hermes-agent/cli.py";

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
      agent: "hermes",
      data: { taskId: task.id, prompt: task.prompt },
    };

    // Use Hermes in single-query mode (--query flag)
    const proc = spawn("python3", [
      this.hermesCliPath,
      "--query", task.prompt,
    ], {
      cwd: this.workDir,
      env: {
        ...process.env,
        // Hermes respects these env vars for configuration
        HERMES_PROFILE: "arena",
      },
    });

    for await (const chunk of proc.stdout) {
      const text = chunk.toString();
      yield {
        type: "text",
        timestamp: timestamp(),
        agent: "hermes",
        data: { text },
      };
    }

    // Capture stderr for errors
    let stderr = "";
    for await (const chunk of proc.stderr) {
      stderr += chunk.toString();
    }

    const exitCode = await new Promise<number>((resolve) => {
      proc.on("close", (code) => resolve(code ?? 1));
    });

    // Hermes writes skills to ~/.hermes/skills/ after learning
    const learned = this.extractLearnedSkills();

    yield {
      type: "completed",
      timestamp: timestamp(),
      agent: "hermes",
      data: {
        success: exitCode === 0,
        summary: exitCode === 0 ? "Task completed" : `Task failed: ${stderr.slice(0, 200)}`,
        tokensUsed: 0,
        knowledgeLearned: learned,
        artifacts: [],
      },
    };
  }

  getLearnedKnowledge(): Knowledge[] {
    return this.extractLearnedSkills();
  }

  private extractLearnedSkills(): Knowledge[] {
    // Hermes stores skills as Markdown files in ~/.hermes/skills/
    const skillsDir = join(process.env.HOME || "/root", ".hermes", "skills");
    if (!existsSync(skillsDir)) return [];

    try {
      const { readdirSync } = require("fs");
      const files = readdirSync(skillsDir) as string[];
      return files
        .filter((f: string) => f.endsWith(".md"))
        .map((f: string) => ({
          id: `hermes-skill-${f}`,
          source: "hermes",
          timestamp: new Date().toISOString(),
          type: "skill" as const,
          content: readFileSync(join(skillsDir, f), "utf-8"),
          confidence: 0.7,
          context: { project: "hermes-learned" },
        }));
    } catch {
      return [];
    }
  }

  ingestKnowledge(knowledge: Knowledge[]): void {
    this.knowledge = [...this.knowledge, ...knowledge];
    // TODO: Write cross-agent knowledge to Hermes memory via CLI
    // hermes memory add "knowledge content"
  }

  async shutdown(): Promise<void> {
    // Hermes CLI exits on its own after --query mode
  }
}
