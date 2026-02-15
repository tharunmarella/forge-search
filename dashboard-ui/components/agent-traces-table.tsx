'use client';

import { useQuery } from '@tanstack/react-query';
import { useRouter } from 'next/navigation';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Eye, Clock, MessageSquare, Wrench, Calendar, Layers } from 'lucide-react';

interface GroupedRun {
  thread_id: string;
  workspace_id: string;
  first_trace_id: string;
  trace_count: number;
  total_duration: number;
  total_messages: number;
  total_tools: number;
  has_plan: boolean;
  status: string;
  first_timestamp: string;
  last_timestamp: string;
}

export function AgentTracesTable() {
  const router = useRouter();
  
  const { data: tracesData, isLoading } = useQuery({
    queryKey: ['traces'],
    queryFn: async () => {
      const res = await fetch('/api/traces?limit=200');
      if (!res.ok) throw new Error('Failed to fetch traces');
      return res.json();
    },
  });

  // Group traces by thread_id to show unique runs
  const groupedRuns: GroupedRun[] = [];
  if (tracesData?.traces) {
    const runsByThread = new Map<string, any[]>();
    
    // Group by thread_id
    tracesData.traces.forEach((trace: any) => {
      const threadId = trace.thread_id;
      if (!runsByThread.has(threadId)) {
        runsByThread.set(threadId, []);
      }
      runsByThread.get(threadId)!.push(trace);
    });
    
    // Create summary for each run
    runsByThread.forEach((traces, threadId) => {
      // Sort by timestamp
      traces.sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());
      
      const firstTrace = traces[0];
      const lastTrace = traces[traces.length - 1];
      
      groupedRuns.push({
        thread_id: threadId,
        workspace_id: firstTrace.workspace_id,
        first_trace_id: firstTrace._id,
        trace_count: traces.length,
        total_duration: traces.reduce((sum, t) => sum + (t.execution_time_ms || 0), 0),
        total_messages: lastTrace.message_count, // Last trace has cumulative count
        total_tools: traces.reduce((sum, t) => sum + (t.tool_call_count || 0), 0),
        has_plan: traces.some(t => t.has_plan),
        status: lastTrace.status,
        first_timestamp: firstTrace.timestamp,
        last_timestamp: lastTrace.timestamp,
      });
    });
    
    // Sort by latest activity
    groupedRuns.sort((a, b) => 
      new Date(b.last_timestamp).getTime() - new Date(a.last_timestamp).getTime()
    );
  }

  const getStatusBadge = (status: string) => {
    if (status === 'success') {
      return <Badge className="bg-green-500/10 text-green-500 hover:bg-green-500/20">Success</Badge>;
    }
    if (status === 'error') {
      return <Badge className="bg-red-500/10 text-red-500 hover:bg-red-500/20">Error</Badge>;
    }
    return <Badge variant="secondary">{status}</Badge>;
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return new Intl.DateTimeFormat('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    }).format(date);
  };

  const formatDuration = (ms: number) => {
    const seconds = Math.floor(ms / 1000);
    if (seconds < 60) return `${seconds}s`;
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}m ${remainingSeconds}s`;
  };

  if (isLoading) {
    return (
      <Card className="mx-4 lg:mx-6">
        <CardHeader>
          <CardTitle>Loading execution runs...</CardTitle>
        </CardHeader>
      </Card>
    );
  }

  return (
    <Card className="mx-4 lg:mx-6">
      <CardHeader>
        <CardTitle>Agent Execution Runs</CardTitle>
        <CardDescription>
          {groupedRuns.length} unique conversation runs
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="rounded-md border">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Workspace</TableHead>
                <TableHead>Turns</TableHead>
                <TableHead>Total Duration</TableHead>
                <TableHead>Messages</TableHead>
                <TableHead>Tools</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Plan</TableHead>
                <TableHead>Started</TableHead>
                <TableHead className="text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {groupedRuns.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={9} className="text-center text-muted-foreground">
                    No execution runs found
                  </TableCell>
                </TableRow>
              ) : (
                groupedRuns.map((run) => (
                  <TableRow key={run.thread_id} className="cursor-pointer hover:bg-muted/50">
                    <TableCell className="font-mono text-sm max-w-[200px] truncate">
                      {run.workspace_id}
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center gap-1">
                        <Layers className="h-3 w-3 text-muted-foreground" />
                        <span className="text-sm font-medium">{run.trace_count}</span>
                        <span className="text-xs text-muted-foreground">turns</span>
                      </div>
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center gap-1">
                        <Clock className="h-3 w-3 text-muted-foreground" />
                        <span className="text-sm">{formatDuration(run.total_duration)}</span>
                      </div>
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center gap-1">
                        <MessageSquare className="h-3 w-3 text-muted-foreground" />
                        <span className="text-sm">{run.total_messages}</span>
                      </div>
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center gap-1">
                        <Wrench className="h-3 w-3 text-muted-foreground" />
                        <span className="text-sm">{run.total_tools}</span>
                      </div>
                    </TableCell>
                    <TableCell>{getStatusBadge(run.status)}</TableCell>
                    <TableCell>
                      {run.has_plan && (
                        <Badge variant="outline" className="text-xs">
                          Multi-step
                        </Badge>
                      )}
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center gap-1 text-sm text-muted-foreground">
                        <Calendar className="h-3 w-3" />
                        {formatDate(run.first_timestamp)}
                      </div>
                    </TableCell>
                    <TableCell className="text-right">
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => router.push(`/run/${run.thread_id}`)}
                      >
                        <Eye className="h-4 w-4 mr-1" />
                        View
                      </Button>
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </div>
      </CardContent>
    </Card>
  );
}
