/**
 * Agent Arena — Agent Adapter Protocol (AAP)
 *
 * The standard interface every agent must implement to plug into the Arena.
 * Agents are disposable; assets are permanent.
 *
 * An agent adapter wraps any AI agent (Ralph, OpenClaw, Hermes, custom)
 * and provides a uniform interface for the Arena to:
 *   - Send tasks with attached assets
 *   - Stream execution events
 *   - Collect learned knowledge
 *   - Score and compare performance
 */

// ─── Core Types ───────────────────────────────────────────────────────

export type AgentCapability =
  | "code"       // Can write/modify code
  | "chat"       // Can converse naturally
  | "browse"     // Can use a web browser
  | "voice"      // Can process/generate speech
  | "design"     // Can work with visual designs
  | "review"     // Can review code/docs
  | "plan"       // Can create plans/specs
  | "research"   // Can search and synthesize information
  | "terminal"   // Can execute shell commands
  | "multi-agent"; // Can spawn sub-agents

export interface AgentIdentity {
  name: string;          // "ralph" | "openclaw" | "hermes" | custom
  version: string;       // Semver
  capabilities: AgentCapability[];
  maxContextTokens: number;
  costPerMToken?: number; // USD per million tokens (for cost-aware routing)
}

// ─── Assets ───────────────────────────────────────────────────────────

/** Reference to an asset in the Arena's Asset Store */
export interface AssetRef {
  type: "skill" | "tool" | "workflow" | "template" | "config";
  path: string;          // Relative to arena/assets/
  version?: string;      // Git SHA or semver
}

/** Reference to an MCP in the Arena's MCP Hub */
export interface McpRef {
  name: string;          // Key from mcp-registry.yml
  capabilities?: string[]; // Subset of capabilities to expose
}

// ─── Knowledge ────────────────────────────────────────────────────────

export interface Knowledge {
  id: string;
  source: string;        // Which agent produced this
  timestamp: string;     // ISO 8601
  type: "memory" | "pattern" | "decision" | "skill";
  content: string;       // The actual knowledge (Markdown)
  confidence: number;    // 0-1, how confident the agent is
  context?: {
    task?: string;       // Task ID that produced this
    project?: string;    // Project context
  };
}

// ─── Tasks ────────────────────────────────────────────────────────────

export interface Constraint {
  type: "time_limit" | "token_budget" | "quality_gate" | "cost_limit";
  value: number;
  unit: string;          // "seconds" | "tokens" | "usd" | "pass_rate"
}

export interface Task {
  id: string;
  prompt: string;
  assets: AssetRef[];    // Skills, tools, templates available for this task
  mcps: McpRef[];        // Which MCPs the agent can use
  constraints: Constraint[];
  workspace: string;     // Filesystem path (arena provides this)
  metadata?: Record<string, unknown>;
}

// ─── Events ───────────────────────────────────────────────────────────

export type EventType =
  | "started"
  | "text"               // Agent produced text output
  | "tool_call"          // Agent called a tool/MCP
  | "tool_result"        // Tool returned a result
  | "knowledge_learned"  // Agent learned something new
  | "error"
  | "completed";

export interface AgentEvent {
  type: EventType;
  timestamp: string;
  agent: string;         // Agent name
  data: unknown;         // Type-specific payload
}

export interface CompletionEvent extends AgentEvent {
  type: "completed";
  data: {
    success: boolean;
    summary: string;
    tokensUsed: number;
    knowledgeLearned: Knowledge[];
    artifacts: string[]; // Files created/modified
  };
}

// ─── Scoring ──────────────────────────────────────────────────────────

export interface ScoreMetric {
  name: string;          // "issues_found", "time_to_complete", etc.
  value: number;
  weight: number;        // 0-1, importance in composite score
}

export interface MatchResult {
  matchId: string;
  task: Task;
  agents: string[];
  scores: Record<string, ScoreMetric[]>; // agent name → metrics
  winner?: string;       // null if tie
  humanVerdict?: string; // Optional human override
  replay: AgentEvent[];  // Full event trace for replay
}

// ─── The Adapter Interface ────────────────────────────────────────────

export interface AgentAdapter {
  /** Agent identity and capabilities */
  identity: AgentIdentity;

  /**
   * Initialize the agent with Arena configuration.
   * Called once when the agent is registered.
   */
  initialize(config: {
    workDir: string;
    mcpHub: McpRef[];
    knowledge: Knowledge[];
  }): Promise<void>;

  /**
   * Execute a task and stream events.
   * The Arena calls this for both solo tasks and competition matches.
   */
  execute(task: Task, context: {
    knowledge: Knowledge[];
    history: AgentEvent[];
  }): AsyncIterable<AgentEvent>;

  /**
   * Return knowledge this agent has accumulated.
   * Called after execution to persist learnings to the Knowledge DB.
   */
  getLearnedKnowledge(): Knowledge[];

  /**
   * Inject knowledge from other agents or the Arena's Knowledge DB.
   * Called before execution to give this agent cross-agent learnings.
   */
  ingestKnowledge(knowledge: Knowledge[]): void;

  /**
   * Gracefully shut down the agent.
   */
  shutdown(): Promise<void>;
}

// ─── Arena Orchestrator ───────────────────────────────────────────────

export type MatchType =
  | "solo"               // Single agent, measure performance
  | "head-to-head"       // Two agents, same task, blind comparison
  | "collaboration"      // Multiple agents work together
  | "tournament"         // Round-robin across N agents
  | "discovery";         // Try all combinations, find best pairings

export interface MatchConfig {
  id: string;
  type: MatchType;
  task: Task;
  agents: string[];      // Agent names to participate
  scoring: ScoreMetric[];
  sandbox: "docker" | "worktree" | "local";
  timeout: number;       // Seconds
}

/**
 * The Arena Orchestrator manages agent registration, match execution,
 * and the global leaderboard.
 */
export interface ArenaOrchestrator {
  /** Register an agent adapter */
  registerAgent(adapter: AgentAdapter): void;

  /** Remove an agent (assets and knowledge are preserved) */
  removeAgent(name: string): void;

  /** List all registered agents */
  listAgents(): AgentIdentity[];

  /** Run a match */
  runMatch(config: MatchConfig): Promise<MatchResult>;

  /** Get the leaderboard */
  getLeaderboard(): Record<string, { elo: number; wins: number; losses: number }>;

  /** Get all knowledge in the Knowledge DB */
  getKnowledge(filter?: { agent?: string; type?: string }): Knowledge[];
}
