export interface Trace {
  _id: string;
  thread_id: string;
  workspace_id: string;
  user_email: string;
  timestamp: string;
  execution_time_ms: number;
  status: string;
  error: string | null;
  message_count: number;
  has_plan: boolean;
  tool_call_count: number;
}

export interface TraceDetail {
  _id: string;
  thread_id: string;
  workspace_id: string;
  timestamp: string;
  request: {
    question: string | null;
    tool_results?: any[];
    attached_files: number;
    attached_images: number;
    is_continuation: boolean;
  };
  response: {
    answer: string | null;
    tool_calls: ToolCall[];
    plan_steps: PlanStep[];
    current_step: number;
    message_count: number;
    messages: Message[];
  };
  execution_time_ms: number;
  status: string;
  error: string | null;
}

export interface ToolCall {
  name: string;
  args: any;
  id: string;
}

export interface PlanStep {
  number: number;
  description: string;
  status: 'pending' | 'in_progress' | 'done';
}

export interface Message {
  type: 'human' | 'ai' | 'tool' | 'system';
  content: string;
  tool_calls?: ToolCall[];
  tool_call_id?: string;
}

export interface FlowNode {
  id: string;
  type: 'input' | 'agent' | 'tool' | 'output';
  data: any;
  position: { x: number; y: number };
}

export interface FlowEdge {
  id: string;
  source: string;
  target: string;
  type: string;
  animated?: boolean;
}

export interface FlowData {
  nodes: FlowNode[];
  edges: FlowEdge[];
  metadata: {
    trace_id: string;
    thread_id: string;
    workspace_id: string;
    timestamp: string;
    execution_time_ms: number;
    status: string;
    total_nodes: number;
    total_edges: number;
  };
}
