'use client';

import { useQuery } from '@tanstack/react-query';
import { useParams, useRouter } from 'next/navigation';
import { useState } from 'react';
import { AppSidebar } from "@/components/app-sidebar"
import { SiteHeader } from "@/components/site-header"
import {
  SidebarInset,
  SidebarProvider,
} from "@/components/ui/sidebar"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { ArrowLeft, Clock, MessageSquare, Wrench, Calendar, Layers, Activity, Copy, ArrowDown, User, Sparkles, WrenchIcon, Download } from 'lucide-react';
import { Skeleton } from '@/components/ui/skeleton';
import { toast } from 'sonner';

export default function RunPage() {
  const params = useParams();
  const router = useRouter();
  const threadId = params.thread_id as string;
  const [viewMode, setViewMode] = useState<'graph' | 'turns'>('graph');

  // Fetch all traces for this thread
  const { data: tracesData, isLoading } = useQuery({
    queryKey: ['run', threadId],
    queryFn: async () => {
      const res = await fetch('/api/traces?limit=500');
      if (!res.ok) throw new Error('Failed to fetch traces');
      const data = await res.json();
      // Filter traces for this thread
      const threadTraces = data.traces.filter((t: any) => t.thread_id === threadId);
      // Sort by timestamp
      threadTraces.sort((a: any, b: any) => 
        new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
      );
      return threadTraces;
    },
    enabled: !!threadId,
  });

  const traces = tracesData || [];
  const firstTrace = traces[0];
  const lastTrace = traces[traces.length - 1];
  
  // Fetch the FULL last trace with message details
  const { data: lastTraceDetail } = useQuery({
    queryKey: ['trace-detail', lastTrace?._id],
    queryFn: async () => {
      if (!lastTrace?._id) return null;
      const res = await fetch(`/api/traces/${lastTrace._id}`);
      if (!res.ok) throw new Error('Failed to fetch trace detail');
      return res.json();
    },
    enabled: !!lastTrace?._id,
  });
  
  const totalDuration = traces.reduce((sum: number, t: any) => sum + (t.execution_time_ms || 0), 0);
  const totalTools = traces.reduce((sum: number, t: any) => sum + (t.tool_call_count || 0), 0);

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  // Message Flow Visualization - shows actual turn-by-turn conversation
  const MessageFlowVisualization = ({ trace }: { trace: any }) => {
    const messages = trace.response.messages;
    const [expandedMessages, setExpandedMessages] = useState<Set<number>>(new Set());
    const traceTimestamp = trace.timestamp || new Date().toISOString();
    
    if (!messages || messages.length === 0) {
      return (
        <div className="text-center py-8 text-muted-foreground">
          No messages to visualize
        </div>
      );
    }

    // Build tool call map from all AI messages
    const toolCallMap = new Map();
    messages.forEach((msg: any) => {
      if (msg.type === 'ai' && msg.tool_calls) {
        msg.tool_calls.forEach((tc: any) => {
          toolCallMap.set(tc.id, tc);
        });
      }
    });

    const toggleExpanded = (index: number) => {
      const newExpanded = new Set(expandedMessages);
      if (newExpanded.has(index)) {
        newExpanded.delete(index);
      } else {
        newExpanded.add(index);
      }
      setExpandedMessages(newExpanded);
    };

    const copyToClipboard = (text: string) => {
      navigator.clipboard.writeText(text);
    };

    const formatTimestamp = (isoString: string) => {
      const date = new Date(isoString);
      return date.toLocaleString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: true
      });
    };

    const exportToMarkdown = () => {
      let markdown = `# Agent Execution Trace\n\n`;
      markdown += `**Thread ID:** ${threadId}\n\n`;
      markdown += `**Timestamp:** ${formatTimestamp(traceTimestamp)}\n\n`;
      markdown += `**Total Messages:** ${messages.length}\n\n`;
      markdown += `---\n\n`;

      messages.forEach((msg: any, index: number) => {
        const nodeNumber = index + 1;
        
        if (msg.type === 'human') {
          markdown += `## ${nodeNumber}. Human Message\n\n`;
          markdown += `${msg.content}\n\n`;
        } else if (msg.type === 'ai') {
          markdown += `## ${nodeNumber}. AI Response\n\n`;
          if (msg.content && msg.content.trim()) {
            markdown += `${msg.content}\n\n`;
          }
          if (msg.tool_calls && msg.tool_calls.length > 0) {
            markdown += `**Tool Calls:**\n`;
            msg.tool_calls.forEach((tc: any) => {
              markdown += `- ${tc.name}\n`;
            });
            markdown += `\n`;
          }
        } else if (msg.type === 'tool') {
          const toolCall = toolCallMap.get(msg.tool_call_id);
          const toolName = toolCall?.name || 'unknown';
          markdown += `## ${nodeNumber}. Tool: ${toolName}\n\n`;
          markdown += `\`\`\`\n${msg.content}\n\`\`\`\n\n`;
        }
        
        markdown += `---\n\n`;
      });

      navigator.clipboard.writeText(markdown).then(() => {
        toast.success('Copied to clipboard!', {
          description: 'Agent trace markdown has been copied to your clipboard.'
        });
      }).catch((err) => {
        toast.error('Failed to copy', {
          description: 'Could not copy to clipboard. Please try again.'
        });
      });
    };

    return (
      <div className="space-y-3 py-4">
        {/* Timestamp Header */}
        <div className="flex items-center justify-between mb-4 px-2">
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <Clock className="h-3 w-3" />
            <span>{formatTimestamp(traceTimestamp)}</span>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={exportToMarkdown}
            className="gap-2"
          >
            <Copy className="h-4 w-4" />
            Copy as Markdown
          </Button>
        </div>

        {messages.map((msg: any, index: number) => {
          const isLast = index === messages.length - 1;
          const isExpanded = expandedMessages.has(index);
          const nodeNumber = index + 1;
          
          // Human Message
          if (msg.type === 'human') {
            const content = msg.content || '';
            const shouldTruncate = content.length > 200;
            const displayContent = isExpanded ? content : content.slice(0, 200);
            
            return (
              <div key={index} className="flex flex-col items-center">
                <div className="w-full p-4 rounded-lg border bg-card">
                  <div className="flex items-center gap-3 mb-2">
                    <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-blue-500/10 text-xs font-medium text-blue-600">
                      {nodeNumber}
                    </div>
                    <User className="h-4 w-4 text-blue-600" />
                    <div className="font-medium text-sm text-blue-600">human</div>
                    <div className="ml-auto flex items-center gap-2">
                      <div className="text-xs text-muted-foreground flex items-center gap-1">
                        <Clock className="h-3 w-3" />
                        <span>{formatTimestamp(traceTimestamp)}</span>
                      </div>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-6 px-2"
                        onClick={() => copyToClipboard(content)}
                      >
                        <Copy className="h-3 w-3" />
                      </Button>
                    </div>
                  </div>
                  <div className="text-sm pl-9 whitespace-pre-wrap text-muted-foreground">
                    {displayContent}
                    {shouldTruncate && !isExpanded && '...'}
                  </div>
                  {shouldTruncate && (
                    <Button
                      variant="ghost"
                      size="sm"
                      className="mt-2 ml-9 h-7 text-xs"
                      onClick={() => toggleExpanded(index)}
                    >
                      {isExpanded ? 'Show Less' : `Show More (${content.length - 200} more characters)`}
                    </Button>
                  )}
                </div>
                {!isLast && <ArrowDown className="h-5 w-5 my-2 text-muted-foreground" />}
              </div>
            );
          }
          
          // AI Message
          if (msg.type === 'ai') {
            const hasToolCalls = msg.tool_calls && msg.tool_calls.length > 0;
            const content = msg.content || '';
            const hasContent = content.trim().length > 0;
            const shouldTruncate = content.length > 200;
            const displayContent = isExpanded ? content : content.slice(0, 200);
            
            return (
              <div key={index} className="flex flex-col items-center">
                <div className="w-full p-4 rounded-lg border bg-card">
                  <div className="flex items-center gap-3 mb-2">
                    <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-purple-500/10 text-xs font-medium text-purple-600">
                      {nodeNumber}
                    </div>
                    <Sparkles className="h-4 w-4 text-purple-600" />
                    <div className="font-medium text-sm text-purple-600">ai</div>
                    {hasToolCalls && (
                      <Badge variant="secondary" className="text-xs">
                        {msg.tool_calls.length} tool call{msg.tool_calls.length > 1 ? 's' : ''}
                      </Badge>
                    )}
                    <div className="ml-auto flex items-center gap-2">
                      <div className="text-xs text-muted-foreground flex items-center gap-1">
                        <Clock className="h-3 w-3" />
                        <span>{formatTimestamp(traceTimestamp)}</span>
                      </div>
                      {hasContent && (
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-6 px-2"
                          onClick={() => copyToClipboard(content)}
                        >
                          <Copy className="h-3 w-3" />
                        </Button>
                      )}
                    </div>
                  </div>
                  
                  {hasContent ? (
                    <>
                      <div className="text-sm pl-9 mb-2 whitespace-pre-wrap">
                        {displayContent}
                        {shouldTruncate && !isExpanded && '...'}
                      </div>
                      {shouldTruncate && (
                        <Button
                          variant="ghost"
                          size="sm"
                          className="ml-9 h-7 text-xs"
                          onClick={() => toggleExpanded(index)}
                        >
                          {isExpanded ? 'Show Less' : `Show More (${content.length - 200} more characters)`}
                        </Button>
                      )}
                    </>
                  ) : hasToolCalls ? (
                    <div className="text-sm pl-9 mb-2 text-muted-foreground italic">
                      Agent executing tools without explanation...
                    </div>
                  ) : null}
                  
                  {hasToolCalls && (
                    <div className="pl-9 mt-2">
                      <div className="text-xs font-medium mb-1">Tool Calls:</div>
                      <div className="flex flex-wrap gap-2">
                        {msg.tool_calls.map((tc: any) => (
                          <Badge key={tc.id} variant="outline" className="font-mono text-xs">
                            {tc.name}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
                {!isLast && <ArrowDown className="h-5 w-5 my-2 text-muted-foreground" />}
              </div>
            );
          }
          
          // Tool Message
          if (msg.type === 'tool') {
            const toolCall = toolCallMap.get(msg.tool_call_id);
            const toolName = toolCall?.name || 'unknown';
            
            // Extract filename from tool arguments if available
            let displayName = toolName;
            if (toolCall?.args) {
              try {
                const args = typeof toolCall.args === 'string' ? JSON.parse(toolCall.args) : toolCall.args;
                // Common patterns for file paths in different tools
                const filePath = args.path || args.file || args.filepath || args.file_path;
                if (filePath) {
                  // Show just the filename, or last 2 parts of path
                  const parts = filePath.split('/');
                  displayName = parts.length > 2 ? `${parts[parts.length - 2]}/${parts[parts.length - 1]}` : parts[parts.length - 1];
                }
              } catch (e) {
                // If parsing fails, just use the tool name
              }
            }
            
            const content = msg.content || '';
            const hasError = content.includes('Error') || content.includes('Failed');
            const shouldTruncate = content.length > 500;
            const displayContent = isExpanded ? content : content.slice(0, 500);
            
            return (
              <div key={index} className="flex flex-col items-center">
                <div className="w-full p-4 rounded-lg border bg-card">
                  <div className="flex items-center gap-3 mb-2">
                    <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-green-500/10 text-xs font-medium text-green-600">
                      {nodeNumber}
                    </div>
                    <WrenchIcon className="h-4 w-4 text-green-600" />
                    <div className="font-medium text-sm font-mono text-green-600">{displayName}</div>
                    <Badge variant="outline" className="text-xs">
                      {toolName}
                    </Badge>
                    {hasError && (
                      <Badge variant="destructive" className="text-xs">
                        error
                      </Badge>
                    )}
                    <div className="ml-auto flex items-center gap-2">
                      <div className="text-xs text-muted-foreground flex items-center gap-1">
                        <Clock className="h-3 w-3" />
                        <span>{formatTimestamp(traceTimestamp)}</span>
                      </div>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-6 px-2"
                        onClick={() => copyToClipboard(content)}
                      >
                        <Copy className="h-3 w-3" />
                      </Button>
                    </div>
                  </div>
                  <div className="text-xs pl-9 font-mono bg-muted/50 p-3 rounded whitespace-pre-wrap overflow-auto">
                    {displayContent}
                    {shouldTruncate && !isExpanded && '...'}
                  </div>
                  {shouldTruncate && (
                    <Button
                      variant="ghost"
                      size="sm"
                      className="mt-2 ml-9 h-7 text-xs"
                      onClick={() => toggleExpanded(index)}
                    >
                      {isExpanded ? 'Show Less' : `Show More (${content.length - 500} more characters)`}
                    </Button>
                  )}
                </div>
                {!isLast && <ArrowDown className="h-5 w-5 my-2 text-muted-foreground" />}
              </div>
            );
          }
          
          return null;
        })}
      </div>
    );
  };

  const formatDuration = (ms: number) => {
    const seconds = Math.floor(ms / 1000);
    if (seconds < 60) return `${seconds}s`;
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}m ${remainingSeconds}s`;
  };

  return (
    <SidebarProvider>
      <AppSidebar variant="inset" />
      <SidebarInset>
        <SiteHeader />
        <div className="flex flex-1 flex-col gap-4 p-4">
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="sm" onClick={() => router.push('/dashboard')}>
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back to Dashboard
            </Button>
          </div>

          {isLoading ? (
            <div className="space-y-4">
              <Skeleton className="h-32 w-full" />
              <Skeleton className="h-96 w-full" />
            </div>
          ) : traces.length > 0 ? (
            <div className="space-y-4">
              {/* Run Header */}
              <Card>
                <CardHeader>
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <CardTitle className="text-2xl">Agent Execution Run</CardTitle>
                      <CardDescription className="mt-2 space-y-2">
                        <div className="font-mono text-sm">{firstTrace?.workspace_id}</div>
                        <div className="flex items-center gap-2">
                          <span className="text-xs text-muted-foreground">Thread ID:</span>
                          <code className="text-xs bg-muted px-2 py-1 rounded font-mono select-all">
                            {threadId}
                          </code>
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-6 px-2"
                            onClick={() => {
                              navigator.clipboard.writeText(threadId);
                            }}
                          >
                            <Copy className="h-3 w-3" />
                          </Button>
                        </div>
                      </CardDescription>
                    </div>
                    <Badge variant={lastTrace?.status === 'success' ? 'default' : 'secondary'}>
                      {lastTrace?.status}
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="flex items-center gap-2">
                      <Layers className="h-4 w-4 text-muted-foreground" />
                      <div>
                        <div className="text-sm font-medium">{traces.length} turns</div>
                        <div className="text-xs text-muted-foreground">Agent Iterations</div>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <Clock className="h-4 w-4 text-muted-foreground" />
                      <div>
                        <div className="text-sm font-medium">{formatDuration(totalDuration)}</div>
                        <div className="text-xs text-muted-foreground">Total Duration</div>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <MessageSquare className="h-4 w-4 text-muted-foreground" />
                      <div>
                        <div className="text-sm font-medium">{lastTrace?.message_count || 0}</div>
                        <div className="text-xs text-muted-foreground">Total Messages</div>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <Wrench className="h-4 w-4 text-muted-foreground" />
                      <div>
                        <div className="text-sm font-medium">{totalTools}</div>
                        <div className="text-xs text-muted-foreground">Tool Calls</div>
                      </div>
                    </div>
                  </div>
                  <div className="mt-4 pt-4 border-t">
                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                      <Calendar className="h-4 w-4" />
                      <span>Started: {formatDate(firstTrace?.timestamp)}</span>
                    </div>
                    {lastTrace && firstTrace?._id !== lastTrace._id && (
                      <div className="flex items-center gap-2 text-sm text-muted-foreground mt-1">
                        <Calendar className="h-4 w-4" />
                        <span>Last activity: {formatDate(lastTrace.timestamp)}</span>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>

              {/* Plan Steps if available */}
              {lastTrace?.response?.plan_steps && lastTrace.response.plan_steps.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle>Execution Plan</CardTitle>
                    <CardDescription>
                      Step {lastTrace.response.current_step} of {lastTrace.response.plan_steps.length}
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      {lastTrace.response.plan_steps.map((step: any) => (
                        <div
                          key={step.number}
                          className="flex items-start gap-3 p-3 rounded-lg border"
                        >
                          <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full border">
                            {step.number}
                          </div>
                          <div className="flex-1">
                            <div className="font-medium">{step.description}</div>
                          </div>
                          <Badge
                            variant={
                              step.status === 'done'
                                ? 'default'
                                : step.status === 'in_progress'
                                ? 'secondary'
                                : 'outline'
                            }
                          >
                            {step.status}
                          </Badge>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Execution Timeline with View Selector */}
              <Card>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div>
                      <CardTitle>Execution Details</CardTitle>
                      <CardDescription>{traces.length} agent turns</CardDescription>
                    </div>
                    <Select value={viewMode} onValueChange={(value: 'graph' | 'turns') => setViewMode(value)}>
                      <SelectTrigger className="w-[180px]">
                        <SelectValue placeholder="Select view" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="graph">Graph View</SelectItem>
                        <SelectItem value="turns">Timeline View</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </CardHeader>
                <CardContent>
                  {viewMode === 'graph' ? (
                    /* Show the message flow visualization */
                    lastTraceDetail?.response?.messages ? (
                      <MessageFlowVisualization trace={lastTraceDetail} />
                    ) : (
                      <div className="text-center py-8 text-muted-foreground">
                        <div className="animate-pulse">Loading latest turn data...</div>
                      </div>
                    )
                  ) : (
                    /* Show turn-by-turn timeline */
                    <div className="space-y-3">
                      {traces.map((trace: any, idx: number) => (
                        <div
                          key={trace._id}
                          className="flex items-start gap-4 p-4 rounded-lg border hover:bg-muted/50 transition-colors"
                        >
                          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary/10 text-primary font-medium text-sm">
                            {idx + 1}
                          </div>
                          <div className="flex-1 space-y-2">
                            <div className="flex items-center justify-between">
                              <div className="flex items-center gap-2">
                                <Activity className="h-4 w-4 text-muted-foreground" />
                                <span className="font-medium text-sm">Turn {idx + 1}</span>
                                <Badge variant="outline" className="text-xs">{trace.status}</Badge>
                              </div>
                              <div className="flex items-center gap-4 text-xs text-muted-foreground">
                                <span className="flex items-center gap-1">
                                  <Clock className="h-3 w-3" />
                                  {Math.round(trace.execution_time_ms)}ms
                                </span>
                                <span className="flex items-center gap-1">
                                  <MessageSquare className="h-3 w-3" />
                                  {trace.message_count} msgs
                                </span>
                                <span className="flex items-center gap-1">
                                  <Wrench className="h-3 w-3" />
                                  {trace.tool_call_count} tools
                                </span>
                              </div>
                            </div>
                            <div className="text-xs text-muted-foreground">
                              {formatDate(trace.timestamp)}
                            </div>
                            {trace.has_plan && (
                              <Badge variant="outline" className="text-xs">
                                With Plan
                              </Badge>
                            )}
                          </div>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => router.push(`/trace/${trace._id}`)}
                          >
                            View Details
                          </Button>
                        </div>
                      ))}
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          ) : (
            <Card>
              <CardContent className="p-6 text-center text-muted-foreground">
                No traces found for this run
              </CardContent>
            </Card>
          )}
        </div>
      </SidebarInset>
    </SidebarProvider>
  );
}
