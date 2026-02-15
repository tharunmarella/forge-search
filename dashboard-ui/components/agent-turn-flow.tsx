'use client';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ArrowRight, ArrowDown } from 'lucide-react';

interface AgentTurnFlowProps {
  trace: any;
}

export function AgentTurnFlow({ trace }: AgentTurnFlowProps) {
  if (!trace?.response?.messages) {
    return null;
  }

  const messages = trace.response.messages;
  
  // Find the last AI message with tool calls and subsequent tool results
  let lastAIMessage = null;
  let lastAIIndex = -1;
  
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].type === 'ai' && messages[i].tool_calls && messages[i].tool_calls.length > 0) {
      lastAIMessage = messages[i];
      lastAIIndex = i;
      break;
    }
  }

  if (!lastAIMessage) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Agent Loop Visualization</CardTitle>
          <CardDescription>No recent tool executions to visualize</CardDescription>
        </CardHeader>
      </Card>
    );
  }

  // Get tool results that follow this AI message
  const toolResults: any[] = [];
  for (let i = lastAIIndex + 1; i < messages.length && messages[i].type === 'tool'; i++) {
    toolResults.push(messages[i]);
  }

  // Build tool call map
  const toolCallMap = new Map();
  lastAIMessage.tool_calls.forEach((tc: any) => {
    toolCallMap.set(tc.id, tc);
  });

  return (
    <Card>
      <CardHeader>
        <CardTitle>Latest Turn: Agent Loop Visualization</CardTitle>
        <CardDescription>
          Block flow diagram showing enrich → agent → tools → routing
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* ENRICH NODE */}
          <div className="flex flex-col items-center">
            <div className="w-full max-w-2xl p-6 rounded-lg border-2 border-blue-500 bg-blue-500/10">
              <div className="flex items-center gap-3 mb-2">
                <div className="text-2xl">🔍</div>
                <div>
                  <div className="font-bold text-lg">ENRICH NODE</div>
                  <div className="text-sm text-muted-foreground">Pre-enrichment context retrieval</div>
                </div>
              </div>
              <div className="mt-3 text-xs space-y-1 text-muted-foreground">
                <div>✓ Semantic search from pgvector</div>
                <div>✓ Architecture map loaded</div>
                <div>✓ Context injected into state</div>
              </div>
            </div>
            <ArrowDown className="h-8 w-8 my-2 text-muted-foreground" />
          </div>

          {/* AGENT NODE */}
          <div className="flex flex-col items-center">
            <div className="w-full max-w-2xl p-6 rounded-lg border-2 border-purple-500 bg-purple-500/10">
              <div className="flex items-center gap-3 mb-2">
                <div className="text-2xl">🤖</div>
                <div>
                  <div className="font-bold text-lg">AGENT NODE (LLM)</div>
                  <div className="text-sm text-muted-foreground">
                    {lastAIMessage.content || 'Reasoning and decision making'}
                  </div>
                </div>
              </div>
              {lastAIMessage.content && (
                <div className="mt-3 p-3 bg-background/50 rounded border text-sm">
                  {lastAIMessage.content.slice(0, 200)}
                  {lastAIMessage.content.length > 200 && '...'}
                </div>
              )}
              <div className="mt-3">
                <div className="text-xs font-medium mb-2">Decided to call {lastAIMessage.tool_calls.length} tool(s):</div>
                <div className="flex flex-wrap gap-2">
                  {lastAIMessage.tool_calls.map((tc: any) => (
                    <Badge key={tc.id} variant="secondary" className="font-mono text-xs">
                      {tc.name}
                    </Badge>
                  ))}
                </div>
              </div>
            </div>
            <ArrowDown className="h-8 w-8 my-2 text-muted-foreground" />
          </div>

          {/* ROUTER */}
          <div className="flex flex-col items-center">
            <div className="w-full max-w-2xl p-4 rounded-lg border-2 border-orange-500 bg-orange-500/10">
              <div className="flex items-center gap-3">
                <div className="text-2xl">🔀</div>
                <div>
                  <div className="font-bold">ROUTER</div>
                  <div className="text-xs text-muted-foreground">
                    Routing based on tool types...
                  </div>
                </div>
              </div>
            </div>
            <ArrowDown className="h-8 w-8 my-2 text-muted-foreground" />
          </div>

          {/* TOOLS EXECUTION */}
          <div className="flex flex-col items-center">
            <div className="w-full max-w-2xl space-y-3">
              <div className="text-center text-sm font-medium text-muted-foreground mb-4">
                Tool Execution
              </div>
              {toolResults.map((toolMsg: any) => {
                const toolCall = Array.from(toolCallMap.values()).find(
                  (tc: any) => tc.id === toolMsg.tool_call_id
                );
                const toolName = toolCall?.name || 'unknown';
                const hasError = toolMsg.content?.includes('Error') || toolMsg.content?.includes('Failed');

                return (
                  <div
                    key={toolMsg.tool_call_id}
                    className={`p-4 rounded-lg border-2 ${
                      hasError
                        ? 'border-red-500 bg-red-500/10'
                        : 'border-green-500 bg-green-500/10'
                    }`}
                  >
                    <div className="flex items-center gap-3 mb-2">
                      <div className="text-xl">{hasError ? '❌' : '⚙️'}</div>
                      <div className="flex-1">
                        <div className="font-bold font-mono">{toolName}</div>
                        <div className="text-xs text-muted-foreground">
                          {hasError ? 'Failed' : 'Success'}
                        </div>
                      </div>
                      <Badge variant={hasError ? 'destructive' : 'default'} className="text-xs">
                        {toolMsg.tool_call_id.slice(-8)}
                      </Badge>
                    </div>
                    <div className="mt-2 p-2 bg-background/50 rounded text-xs font-mono max-h-20 overflow-hidden">
                      {toolMsg.content.slice(0, 150)}
                      {toolMsg.content.length > 150 && '...'}
                    </div>
                  </div>
                );
              })}
            </div>
            <ArrowDown className="h-8 w-8 my-2 text-muted-foreground" />
          </div>

          {/* NEXT STATE */}
          <div className="flex flex-col items-center">
            <div className="w-full max-w-2xl p-6 rounded-lg border-2 border-cyan-500 bg-cyan-500/10">
              <div className="flex items-center gap-3">
                <div className="text-2xl">🔄</div>
                <div>
                  <div className="font-bold text-lg">LOOP BACK TO AGENT</div>
                  <div className="text-sm text-muted-foreground">
                    Tool results added to state → Agent reasons again
                  </div>
                </div>
              </div>
              <div className="mt-3 text-xs text-muted-foreground">
                Or pause/end if IDE tools or no more actions needed
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
