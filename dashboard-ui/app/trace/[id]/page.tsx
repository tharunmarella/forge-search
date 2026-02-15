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
import { ArrowLeft, Clock, MessageSquare, Wrench, Calendar, ChevronDown, ChevronUp, Copy } from 'lucide-react';
import { Skeleton } from '@/components/ui/skeleton';

export default function TracePage() {
  const params = useParams();
  const router = useRouter();
  const traceId = params.id as string;
  const [expandedMessages, setExpandedMessages] = useState<Set<number>>(new Set());

  const { data: traceData, isLoading } = useQuery({
    queryKey: ['trace', traceId],
    queryFn: async () => {
      const res = await fetch(`/api/traces/${traceId}`);
      if (!res.ok) throw new Error('Failed to fetch trace');
      return res.json();
    },
    enabled: !!traceId,
  });

  const { data: flowData } = useQuery({
    queryKey: ['flow', traceId],
    queryFn: async () => {
      const res = await fetch(`/api/traces/${traceId}/flow`);
      if (!res.ok) throw new Error('Failed to fetch flow');
      return res.json();
    },
    enabled: !!traceId,
  });

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  const toggleExpanded = (idx: number) => {
    const newExpanded = new Set(expandedMessages);
    if (newExpanded.has(idx)) {
      newExpanded.delete(idx);
    } else {
      newExpanded.add(idx);
    }
    setExpandedMessages(newExpanded);
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  // Build a map of tool_call_id -> tool_name
  const toolCallMap = new Map<string, string>();
  if (traceData?.response?.messages) {
    traceData.response.messages.forEach((msg: any) => {
      if (msg.type === 'ai' && msg.tool_calls) {
        msg.tool_calls.forEach((tc: any) => {
          toolCallMap.set(tc.id, tc.name);
        });
      }
    });
  }

  const renderMessageContent = (msg: any, idx: number) => {
    const content = typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content, null, 2);
    const isLong = content.length > 500;
    const isExpanded = expandedMessages.has(idx);
    const displayContent = isLong && !isExpanded ? content.slice(0, 500) + '...' : content;

    return (
      <div className="space-y-2">
        <div className="relative">
          <pre className="text-sm whitespace-pre-wrap font-mono bg-background/50 p-3 rounded border max-h-[400px] overflow-y-auto">
            {displayContent}
          </pre>
          <Button
            variant="ghost"
            size="sm"
            className="absolute top-2 right-2"
            onClick={() => copyToClipboard(content)}
          >
            <Copy className="h-3 w-3" />
          </Button>
        </div>
        {isLong && (
          <Button
            variant="outline"
            size="sm"
            onClick={() => toggleExpanded(idx)}
            className="w-full"
          >
            {isExpanded ? (
              <>
                <ChevronUp className="h-4 w-4 mr-2" />
                Show Less
              </>
            ) : (
              <>
                <ChevronDown className="h-4 w-4 mr-2" />
                Show More ({content.length - 500} more characters)
              </>
            )}
          </Button>
        )}
      </div>
    );
  };

  return (
    <SidebarProvider>
      <AppSidebar variant="inset" />
      <SidebarInset>
        <SiteHeader />
        <div className="flex flex-1 flex-col gap-4 p-4">
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="sm" onClick={() => router.back()}>
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back
            </Button>
          </div>

          {isLoading ? (
            <div className="space-y-4">
              <Skeleton className="h-32 w-full" />
              <Skeleton className="h-96 w-full" />
            </div>
          ) : traceData ? (
            <div className="space-y-4">
              {/* Trace Header */}
              <Card>
                <CardHeader>
                  <div className="flex items-start justify-between">
                    <div>
                      <CardTitle className="text-2xl">Execution Trace</CardTitle>
                      <CardDescription className="mt-2">
                        {traceData.workspace_id} • Thread: {traceData.thread_id}
                      </CardDescription>
                    </div>
                    <Badge variant={traceData.status === 'success' ? 'default' : 'destructive'}>
                      {traceData.status}
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="flex items-center gap-2">
                      <Clock className="h-4 w-4 text-muted-foreground" />
                      <div>
                        <div className="text-sm font-medium">{Math.round(traceData.execution_time_ms)}ms</div>
                        <div className="text-xs text-muted-foreground">Execution Time</div>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <MessageSquare className="h-4 w-4 text-muted-foreground" />
                      <div>
                        <div className="text-sm font-medium">{traceData.response.message_count}</div>
                        <div className="text-xs text-muted-foreground">Messages</div>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <Wrench className="h-4 w-4 text-muted-foreground" />
                      <div>
                        <div className="text-sm font-medium">{traceData.response.tool_calls?.length || 0}</div>
                        <div className="text-xs text-muted-foreground">Tool Calls</div>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <Calendar className="h-4 w-4 text-muted-foreground" />
                      <div>
                        <div className="text-sm font-medium">{formatDate(traceData.timestamp)}</div>
                        <div className="text-xs text-muted-foreground">Timestamp</div>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Plan Steps */}
              {traceData.response.plan_steps && traceData.response.plan_steps.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle>Execution Plan</CardTitle>
                    <CardDescription>
                      Step {traceData.response.current_step} of {traceData.response.plan_steps.length}
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      {traceData.response.plan_steps.map((step: any) => (
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

              {/* Flow Graph Info */}
              {flowData && (
                <Card>
                  <CardHeader>
                    <CardTitle>Execution Flow</CardTitle>
                    <CardDescription>
                      {flowData.metadata.total_nodes} nodes • {flowData.metadata.total_edges} edges
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="text-sm text-muted-foreground">
                      Detailed flow visualization coming soon...
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Messages */}
              <Card>
                <CardHeader>
                  <CardTitle>Message History</CardTitle>
                  <CardDescription>{traceData.response.messages?.length || 0} messages</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {traceData.response.messages?.map((msg: any, idx: number) => (
                      <div
                        key={idx}
                        className={`p-4 rounded-lg border ${
                          msg.type === 'human'
                            ? 'bg-blue-500/10 border-blue-500/20'
                            : msg.type === 'ai'
                            ? 'bg-purple-500/10 border-purple-500/20'
                            : msg.type === 'tool'
                            ? 'bg-green-500/10 border-green-500/20'
                            : 'bg-muted'
                        }`}
                      >
                        <div className="flex items-center gap-2 mb-3">
                          <Badge variant="outline" className="text-xs">
                            {msg.type}
                          </Badge>
                          {msg.type === 'tool' && msg.tool_call_id && (
                            <Badge variant="secondary" className="text-xs font-mono">
                              {toolCallMap.get(msg.tool_call_id) || msg.tool_call_id.slice(-8)}
                            </Badge>
                          )}
                          {msg.type === 'ai' && msg.tool_calls && msg.tool_calls.length > 0 && !msg.content && (
                            <Badge variant="secondary" className="text-xs">
                              {msg.tool_calls.length} tool call{msg.tool_calls.length > 1 ? 's' : ''}
                            </Badge>
                          )}
                        </div>
                        {msg.content ? (
                          renderMessageContent(msg, idx)
                        ) : msg.type === 'ai' && msg.tool_calls && msg.tool_calls.length > 0 ? (
                          <div className="text-sm text-muted-foreground italic">
                            Agent executing tools without explanation...
                          </div>
                        ) : (
                          <div className="text-sm text-muted-foreground italic">
                            (Empty message)
                          </div>
                        )}
                        {msg.tool_calls && msg.tool_calls.length > 0 && (
                          <div className="mt-3 space-y-2">
                            <div className="text-xs font-medium text-muted-foreground">Tool Calls:</div>
                            {msg.tool_calls.map((tc: any, tcIdx: number) => (
                              <div key={tcIdx} className="p-3 bg-background rounded border space-y-1">
                                <div className="flex items-center justify-between">
                                  <span className="font-mono text-sm font-medium">{tc.name}</span>
                                  <Badge variant="outline" className="text-xs font-mono">
                                    {tc.id?.slice(-8)}
                                  </Badge>
                                </div>
                                {tc.args && Object.keys(tc.args).length > 0 && (
                                  <pre className="text-xs mt-2 p-2 bg-muted rounded overflow-x-auto">
                                    {JSON.stringify(tc.args, null, 2)}
                                  </pre>
                                )}
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>
          ) : (
            <Card>
              <CardContent className="p-6 text-center text-muted-foreground">
                Trace not found
              </CardContent>
            </Card>
          )}
        </div>
      </SidebarInset>
    </SidebarProvider>
  );
}
