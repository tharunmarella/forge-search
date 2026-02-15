# Agent Trace Enhancements Plan

## Goal
Enhance traces to capture detailed metrics at each step of the agent loop for complete end-to-end understanding.

## Current State
- Basic trace logging in MongoDB with:
  - `thread_id`, `workspace_id`, `timestamp`
  - `execution_time_ms`, `status`, `error`
  - `request` and `response` objects
  - `messages` array

## Enhanced Trace Structure

### Step Metrics to Add

```python
{
    "step_metrics": {
        # Enrichment Phase (Node 0)
        "enrichment": {
            "time_ms": float,
            "context_size_chars": int,
            "context_snippets": int,
            "search_query": str,
            "component_count": int,
            "vector_search_time_ms": float,
            "architecture_map_time_ms": float,
            "skipped": bool,  # If NONE_NEEDED
        },
        
        # Agent Turns (repeated for each LLM call)
        "agent_turns": [
            {
                "turn_number": int,
                "time_ms": float,
                "model_name": str,
                "temperature": float,
                
                # Token usage
                "input_tokens": int,
                "output_tokens": int,
                "cached_tokens": int,  # If prompt caching enabled
                "total_tokens": int,
                
                # Context window
                "input_chars": int,
                "system_prompt_chars": int,
                "enriched_context_chars": int,
                "history_chars": int,
                
                # Tool calls made
                "tool_calls": [
                    {
                        "name": str,
                        "args": dict,  # Tool arguments
                        "result_chars": int,
                        "time_ms": float,
                        "success": bool,
                        "error": str,  # If failed
                    }
                ],
                
                # Routing decision
                "router_decision": str,  # "tools" | "ide" | "end"
                "router_reason": str,
            }
        ],
        
        # Overall metrics
        "total_tokens": int,
        "total_tool_calls": int,
        "total_turns": int,
        "context_switches": int,  # IDE pauses
    }
}
```

## Implementation Steps

### 1. Enhance AgentState (app/core/agent.py)

```python
class AgentState(TypedDict):
    workspace_id: str
    user_email: str
    messages: Annotated[list[BaseMessage], add_messages]
    enriched_context: str
    plan_steps: list[dict]
    current_step: int
    # NEW: Track metrics throughout execution
    step_metrics: dict
```

### 2. Add Timing to Each Node

```python
import time

async def enrich_context(state: AgentState) -> dict:
    start = time.time()
    step_metrics = state.get('step_metrics', {})
    
    # ... existing enrichment logic ...
    
    step_metrics['enrichment'] = {
        'time_ms': (time.time() - start) * 1000,
        'context_size_chars': len(new_context),
        'context_snippets': len(results),
        # ... other metrics ...
    }
    
    return {
        "enriched_context": new_context,
        "step_metrics": step_metrics,
    }
```

### 3. Capture LLM Token Usage

```python
async def call_model(state: AgentState) -> dict:
    start = time.time()
    step_metrics = state.get('step_metrics', {})
    
    # Invoke LLM
    response = await llm_with_tools.ainvoke(messages)
    
    # Extract token usage from response metadata
    if hasattr(response, 'response_metadata'):
        usage = response.response_metadata.get('usage', {})
        turn_metrics = {
            'time_ms': (time.time() - start) * 1000,
            'input_tokens': usage.get('input_tokens', 0),
            'output_tokens': usage.get('output_tokens', 0),
            'total_tokens': usage.get('total_tokens', 0),
            # ... more metrics ...
        }
    
    agent_turns = step_metrics.get('agent_turns', [])
    agent_turns.append(turn_metrics)
    step_metrics['agent_turns'] = agent_turns
    
    return {"messages": [response], "step_metrics": step_metrics}
```

### 4. Track Tool Execution Time

```python
async def execute_server_tools(state: AgentState) -> dict:
    step_metrics = state.get('step_metrics', {})
    agent_turns = step_metrics.get('agent_turns', [])
    
    if agent_turns:
        current_turn = agent_turns[-1]
        tool_results = []
        
        for tool_call in tool_calls:
            start = time.time()
            try:
                result = await tool.ainvoke(tool_call['args'])
                tool_results.append({
                    'name': tool_call['name'],
                    'time_ms': (time.time() - start) * 1000,
                    'result_chars': len(str(result)),
                    'success': True,
                })
            except Exception as e:
                tool_results.append({
                    'name': tool_call['name'],
                    'time_ms': (time.time() - start) * 1000,
                    'success': False,
                    'error': str(e),
                })
        
        current_turn['tool_calls'] = tool_results
```

### 5. Update Trace Logging (app/api/chat.py)

Already updated `_log_trace_to_mongo` to accept `step_metrics` parameter.

Now update all call sites:

```python
await _log_trace_to_mongo(
    thread_id=thread_id,
    workspace_id=workspace_id,
    user_email=user_email,
    request_data=request_data,
    response_data=response_data,
    execution_time_ms=execution_time_ms,
    status="success",
    step_metrics=final_state.get('step_metrics', {}),  # NEW
)
```

## Dashboard Visualization Enhancements

### Update Run Page to Show Step Metrics

In `dashboard-ui/app/run/[thread_id]/page.tsx`:

1. **Enrichment Node** - Show actual metrics:
   ```tsx
   <div className="text-xs pl-9">
     <div>✓ Search query: {enrichment.search_query}</div>
     <div>✓ Snippets: {enrichment.context_snippets}</div>
     <div>✓ Context size: {enrichment.context_size_chars} chars</div>
     <div>✓ Time: {enrichment.time_ms.toFixed(1)}ms</div>
   </div>
   ```

2. **AI Node** - Show token usage:
   ```tsx
   <Badge variant="secondary">
     {turn.total_tokens} tokens ({turn.input_tokens} in / {turn.output_tokens} out)
   </Badge>
   <Badge variant="outline">
     {turn.time_ms.toFixed(0)}ms
   </Badge>
   ```

3. **Tool Nodes** - Show execution time:
   ```tsx
   <div className="text-xs text-muted-foreground">
     Executed in {tool.time_ms.toFixed(1)}ms
   </div>
   ```

4. **Add Metrics Summary Card**:
   ```tsx
   <Card>
     <CardHeader>
       <CardTitle>Performance Metrics</CardTitle>
     </CardHeader>
     <CardContent>
       <div className="grid grid-cols-3 gap-4">
         <MetricItem label="Total Tokens" value={totalTokens} />
         <MetricItem label="Enrichment" value={`${enrichTime}ms`} />
         <MetricItem label="LLM Time" value={`${llmTime}ms`} />
       </div>
     </CardContent>
   </Card>
   ```

## Benefits

1. **Performance Analysis**: See where time is spent (enrichment vs LLM vs tools)
2. **Cost Tracking**: Monitor token usage per conversation
3. **Debugging**: Understand exactly what context was injected and why
4. **Optimization**: Identify slow tools or excessive context retrieval
5. **Quality**: Verify enrichment is working as expected

## Next Steps

1. Implement step_metrics tracking in agent.py
2. Update all trace logging call sites
3. Enhance dashboard to display metrics
4. Add metrics aggregation endpoint for analytics
