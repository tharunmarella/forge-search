'use client';

import { useQuery } from '@tanstack/react-query';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { ChartContainer, ChartTooltip, ChartTooltipContent } from '@/components/ui/chart';
import { Area, AreaChart, CartesianGrid, XAxis, YAxis } from 'recharts';

export function TraceExecutionChart() {
  const { data: tracesData } = useQuery({
    queryKey: ['traces'],
    queryFn: async () => {
      const res = await fetch('/api/traces?limit=50');
      if (!res.ok) throw new Error('Failed to fetch traces');
      return res.json();
    },
  });

  const traces = tracesData?.traces || [];
  
  // Prepare chart data - group by hour
  const chartData = traces
    .slice()
    .reverse()
    .map((trace: any, index: number) => ({
      index: index + 1,
      time: trace.execution_time_ms,
      messages: trace.message_count,
      tools: trace.tool_call_count,
    }));

  return (
    <Card>
      <CardHeader>
        <CardTitle>Execution Performance</CardTitle>
        <CardDescription>
          Execution time and activity for recent traces
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ChartContainer
          config={{
            time: {
              label: 'Execution Time (ms)',
              color: 'hsl(var(--chart-1))',
            },
            messages: {
              label: 'Messages',
              color: 'hsl(var(--chart-2))',
            },
          }}
          className="h-[300px]"
        >
          <AreaChart data={chartData}>
            <defs>
              <linearGradient id="fillTime" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="hsl(var(--chart-1))" stopOpacity={0.8} />
                <stop offset="95%" stopColor="hsl(var(--chart-1))" stopOpacity={0.1} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis
              dataKey="index"
              tickLine={false}
              axisLine={false}
              tickMargin={8}
              label={{ value: 'Trace Number', position: 'insideBottom', offset: -5 }}
            />
            <YAxis
              tickLine={false}
              axisLine={false}
              tickMargin={8}
              label={{ value: 'Time (ms)', angle: -90, position: 'insideLeft' }}
            />
            <ChartTooltip content={<ChartTooltipContent />} />
            <Area
              type="monotone"
              dataKey="time"
              stroke="hsl(var(--chart-1))"
              fillOpacity={1}
              fill="url(#fillTime)"
            />
          </AreaChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}
