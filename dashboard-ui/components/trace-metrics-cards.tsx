'use client';

import { useQuery } from '@tanstack/react-query';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Activity, Clock, CheckCircle2, Layers } from 'lucide-react';

export function TraceMetricsCards() {
  const { data: tracesData } = useQuery({
    queryKey: ['traces'],
    queryFn: async () => {
      const res = await fetch('/api/traces?limit=200');
      if (!res.ok) throw new Error('Failed to fetch traces');
      return res.json();
    },
  });

  const traces = tracesData?.traces || [];
  
  // Group by thread_id to get unique runs
  const uniqueThreads = new Set(traces.map((t: any) => t.thread_id));
  const totalRuns = uniqueThreads.size;
  
  // Calculate metrics per run
  const runMetrics = new Map();
  traces.forEach((trace: any) => {
    const threadId = trace.thread_id;
    if (!runMetrics.has(threadId)) {
      runMetrics.set(threadId, {
        totalTime: 0,
        status: trace.status,
        turnCount: 0,
      });
    }
    const metrics = runMetrics.get(threadId);
    metrics.totalTime += trace.execution_time_ms || 0;
    metrics.status = trace.status; // Use latest status
    metrics.turnCount += 1;
  });
  
  const runArray = Array.from(runMetrics.values());
  const avgTimePerRun = runArray.length > 0
    ? Math.round(runArray.reduce((sum, r) => sum + r.totalTime, 0) / runArray.length)
    : 0;
  const successCount = runArray.filter(r => r.status === 'success').length;
  const avgTurnsPerRun = runArray.length > 0
    ? Math.round(runArray.reduce((sum, r) => sum + r.turnCount, 0) / runArray.length)
    : 0;

  return (
    <div className="grid gap-4 px-4 md:grid-cols-2 lg:grid-cols-4 lg:px-6">
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Total Runs</CardTitle>
          <Activity className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{totalRuns}</div>
          <p className="text-xs text-muted-foreground">
            Unique conversation runs
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Avg Duration</CardTitle>
          <Clock className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">
            {avgTimePerRun < 1000 
              ? `${avgTimePerRun}ms` 
              : `${Math.round(avgTimePerRun / 1000)}s`}
          </div>
          <p className="text-xs text-muted-foreground">
            Per complete run
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Success Rate</CardTitle>
          <CheckCircle2 className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">
            {totalRuns > 0 ? Math.round((successCount / totalRuns) * 100) : 0}%
          </div>
          <p className="text-xs text-muted-foreground">
            {successCount} of {totalRuns} successful
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Avg Turns</CardTitle>
          <Layers className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{avgTurnsPerRun}</div>
          <p className="text-xs text-muted-foreground">
            Agent iterations per run
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
