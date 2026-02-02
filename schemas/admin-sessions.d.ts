/**
 * Toolbridge Admin Sessions API Types
 *
 * TypeScript type definitions for the toolbridge admin sessions API.
 * Copy this file to your client project or use it as a reference.
 *
 * Generated from: schemas/admin-sessions.json
 */

/**
 * The function details within a tool call
 */
export interface ToolCallFunction {
  /** Name of the function being called */
  name: string;
  /** JSON-encoded string of function arguments */
  arguments: string;
}

/**
 * An OpenAI-format tool call (post-transformation)
 */
export interface ToolCall {
  /** Unique identifier for this tool call (e.g., 'call_abc123') */
  id: string;
  /** Type of tool call (always 'function' for now) */
  type: "function";
  /** The function being called */
  function: ToolCallFunction;
}

/**
 * A single message in the chat history
 */
export interface ChatMessage {
  /** Unix timestamp (seconds) when the message was recorded */
  timestamp: number;
  /** Whether this message was part of an incoming request or outgoing response */
  direction: "request" | "response";
  /** The chat role of this message */
  role: "user" | "assistant" | "system" | "tool";
  /** Message content (after transformation if applicable) */
  content: string;
  /** Tool calls if this is an assistant message containing tool invocations */
  tool_calls?: ToolCall[] | null;
  /** Original content before transformation. Only present if the message was transformed. */
  raw_content?: string | null;
  /** Original JSON payload (parsed). Contains the raw message data from the request or response. */
  debug?: Record<string, unknown> | null;
  /**
   * Snapshot of prompt_tokens at the time this message was logged.
   * Represents the total context size including this message.
   * Calculate deltas between consecutive messages to see per-message token cost.
   * The first message's prompt_tokens reveals hidden overhead (system prompt, tools, template).
   */
  prompt_tokens?: number | null;
}

/**
 * Summary statistics for a single session
 */
export interface SessionSummary {
  /** Unix timestamp (seconds) when the session was created */
  created_at: number;
  /** Unix timestamp (seconds) of last activity in the session */
  last_seen_at: number;
  /** Time in seconds since session creation */
  age_seconds: number;
  /** Time in seconds since last activity */
  idle_seconds: number;
  /** Total number of requests processed in this session */
  request_count: number;
  /** Total number of tool calls processed */
  tool_calls_total: number;
  /** Number of tool calls that were transformed/fixed by the proxy */
  tool_calls_fixed: number;
  /** Number of tool calls where transformation failed */
  tool_calls_failed: number;
  /** Ratio of fixed to total tool calls (0.0 to 1.0) */
  fix_rate: number;
  /** Client IP address if available */
  client_ip: string | null;
  /** Prompt tokens from the most recent request (current context window size) */
  last_prompt_tokens: number | null;
  /** Completion tokens from the most recent request */
  last_completion_tokens: number | null;
  /** Total tokens from the most recent request */
  last_total_tokens: number | null;
  /** Cumulative prompt tokens across all requests in this session */
  prompt_tokens_total: number | null;
  /** Cumulative completion tokens across all requests in this session */
  completion_tokens_total: number | null;
  /** Cumulative total tokens across all requests in this session */
  total_tokens_total: number | null;
  /** Detected AI coding assistant name (e.g., 'Cline', 'Cursor', 'Claude Code'). Null if no agent detected. */
  detected_agent: string | null;
  /** Confidence level of agent detection: 'high' for known agent patterns, 'medium' for generic patterns, 'low' for weak signals. Null if no agent detected. */
  detected_agent_confidence: "high" | "medium" | "low" | null;
}

/**
 * Response from GET /admin/sessions - lists all active sessions
 */
export interface SessionListResponse {
  /** Count of currently active sessions */
  active_sessions: number;
  /** Session timeout configuration in seconds (default: 3600) */
  session_timeout_seconds: number;
  /** Map of session_id to session summary */
  sessions: Record<string, SessionSummary>;
}

/**
 * Response from GET /admin/sessions/{session_id}
 * Detailed session information with optional message history
 */
export interface SessionDetailResponse {
  /** The session identifier */
  session_id: string;
  /** Unix timestamp when session was created */
  created_at: number;
  /** Unix timestamp of last activity */
  last_seen_at: number;
  /** Time since session creation */
  age_seconds: number;
  /** Time since last activity */
  idle_seconds: number;
  /** Total requests in this session */
  request_count: number;
  /** Total tool calls processed */
  tool_calls_total: number;
  /** Tool calls that were transformed */
  tool_calls_fixed: number;
  /** Tool calls that failed transformation */
  tool_calls_failed: number;
  /** Ratio of fixed to total tool calls */
  fix_rate: number;
  /** Client IP address if available */
  client_ip: string | null;
  /** Prompt tokens from the most recent request (current context window size) */
  last_prompt_tokens: number | null;
  /** Completion tokens from the most recent request */
  last_completion_tokens: number | null;
  /** Total tokens from the most recent request */
  last_total_tokens: number | null;
  /** Cumulative prompt tokens across all requests in this session */
  prompt_tokens_total: number | null;
  /** Cumulative completion tokens across all requests in this session */
  completion_tokens_total: number | null;
  /** Cumulative total tokens across all requests in this session */
  total_tokens_total: number | null;
  /** Detected AI coding assistant name (e.g., 'Cline', 'Cursor', 'Claude Code'). Null if no agent detected. */
  detected_agent: string | null;
  /** Confidence level of agent detection: 'high' for known agent patterns, 'medium' for generic patterns, 'low' for weak signals. Null if no agent detected. */
  detected_agent_confidence: "high" | "medium" | "low" | null;
  /** Total number of messages in the buffer (only if include_messages=true) */
  total_messages?: number;
  /** Configured message buffer size limit (only if include_messages=true) */
  message_buffer_size?: number;
  /** Message history (only if include_messages=true) */
  messages?: ChatMessage[];
}

/**
 * Error response returned when a request fails
 */
export interface ErrorResponse {
  /** Human-readable error message */
  error: string;
  /** The session ID that was requested (if applicable) */
  session_id?: string;
}

/**
 * Query parameters for GET /admin/sessions/{session_id}
 */
export interface SessionDetailParams {
  /** Include message history in response (default: true) */
  include_messages?: boolean;
  /** Maximum number of messages to return (default: 100) */
  message_limit?: number;
}
