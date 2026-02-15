"""
Traces API — View and visualize agent execution runs.

Endpoints:
  GET /traces              — List all execution traces
  GET /traces/{trace_id}   — Get detailed trace with full execution flow
  GET /traces/dashboard    — Serve the visualization dashboard HTML
"""

import logging
import os
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/traces", tags=["traces"])

# ── MongoDB Connection ──────────────────────────────────────────────
_mongo_url = os.getenv("MONGODB_URL", "")
if _mongo_url:
    try:
        _mongo_client = AsyncIOMotorClient(_mongo_url)
        _mongo_db = _mongo_client["forge_traces"]
        _traces_collection = _mongo_db["traces"]
        logger.info("Traces API: MongoDB connection established")
    except Exception as e:
        logger.error("Traces API: MongoDB init failed: %s", e)
        _mongo_client = None
        _traces_collection = None
else:
    _mongo_client = None
    _traces_collection = None
    logger.warning("Traces API: MongoDB not configured (set MONGODB_URL)")


def _serialize_trace(doc: dict) -> dict:
    """Convert MongoDB document to JSON-safe dict."""
    doc["_id"] = str(doc["_id"])
    if "timestamp" in doc and isinstance(doc["timestamp"], datetime):
        doc["timestamp"] = doc["timestamp"].isoformat()
    return doc


@router.get("")
async def list_traces(
    workspace_id: Optional[str] = None,
    limit: int = 50,
    skip: int = 0
):
    """List execution traces with optional filtering."""
    if _traces_collection is None:
        raise HTTPException(status_code=503, detail="MongoDB not configured")
    
    query = {}
    if workspace_id:
        query["workspace_id"] = workspace_id
    
    cursor = _traces_collection.find(query).sort("timestamp", -1).skip(skip).limit(limit)
    traces = await cursor.to_list(length=limit)
    
    # Return summary info (no full messages)
    result = []
    for trace in traces:
        result.append({
            "_id": str(trace["_id"]),
            "thread_id": trace.get("thread_id"),
            "workspace_id": trace.get("workspace_id"),
            "user_email": trace.get("user_email"),
            "timestamp": trace["timestamp"].isoformat() if isinstance(trace.get("timestamp"), datetime) else trace.get("timestamp"),
            "execution_time_ms": trace.get("execution_time_ms"),
            "status": trace.get("status"),
            "error": trace.get("error"),
            "message_count": trace.get("response", {}).get("message_count", 0),
            "has_plan": bool(trace.get("response", {}).get("plan_steps")),
            "tool_call_count": len(trace.get("response", {}).get("tool_calls") or []),
        })
    
    return {
        "traces": result,
        "count": len(result),
        "has_more": len(result) == limit
    }


@router.get("/{trace_id}")
async def get_trace(trace_id: str):
    """Get full trace details including execution flow."""
    if _traces_collection is None:
        raise HTTPException(status_code=503, detail="MongoDB not configured")
    
    try:
        doc = await _traces_collection.find_one({"_id": ObjectId(trace_id)})
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid trace ID: {e}")
    
    if not doc:
        raise HTTPException(status_code=404, detail="Trace not found")
    
    return _serialize_trace(doc)


@router.delete("/{trace_id}")
async def delete_trace(trace_id: str):
    """Delete a specific trace by ID."""
    if _traces_collection is None:
        raise HTTPException(status_code=503, detail="MongoDB not configured")
    
    try:
        result = await _traces_collection.delete_one({"_id": ObjectId(trace_id)})
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid trace ID: {e}")
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Trace not found")
    
    logger.info(f"Deleted trace: {trace_id}")
    return {"success": True, "deleted_id": trace_id}


@router.delete("/thread/{thread_id}")
async def delete_thread_traces(thread_id: str):
    """Delete all traces for a specific thread (conversation run)."""
    if _traces_collection is None:
        raise HTTPException(status_code=503, detail="MongoDB not configured")
    
    result = await _traces_collection.delete_many({"thread_id": thread_id})
    
    logger.info(f"Deleted {result.deleted_count} traces for thread: {thread_id}")
    return {
        "success": True,
        "deleted_count": result.deleted_count,
        "thread_id": thread_id
    }


@router.get("/{trace_id}/flow")
async def get_trace_flow(trace_id: str):
    """Get trace as a node-edge graph for visualization."""
    if _traces_collection is None:
        raise HTTPException(status_code=503, detail="MongoDB not configured")
    
    try:
        doc = await _traces_collection.find_one({"_id": ObjectId(trace_id)})
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid trace ID: {e}")
    
    if not doc:
        raise HTTPException(status_code=404, detail="Trace not found")
    
    # Build flow graph from messages
    messages = doc.get("response", {}).get("messages", [])
    nodes = []
    edges = []
    
    node_id_counter = 0
    
    # Add initial request node
    request = doc.get("request", {})
    nodes.append({
        "id": f"node_{node_id_counter}",
        "type": "input",
        "data": {
            "label": "User Input",
            "question": request.get("question"),
            "attached_files": request.get("attached_files", 0),
            "attached_images": request.get("attached_images", 0),
            "is_continuation": request.get("is_continuation", False),
            "enriched_context": request.get("enriched_context", ""),
        },
        "position": {"x": 250, "y": 0}
    })
    last_node_id = f"node_{node_id_counter}"
    node_id_counter += 1
    
    # Process messages to build flow
    y_offset = 150
    for i, msg in enumerate(messages):
        msg_type = msg.get("type")
        
        if msg_type == "human":
            nodes.append({
                "id": f"node_{node_id_counter}",
                "type": "input",
                "data": {
                    "label": "User Message",
                    "content": msg.get("content", "")[:200],
                },
                "position": {"x": 250, "y": y_offset}
            })
            edges.append({
                "id": f"edge_{len(edges)}",
                "source": last_node_id,
                "target": f"node_{node_id_counter}",
                "type": "smoothstep"
            })
            last_node_id = f"node_{node_id_counter}"
            node_id_counter += 1
            y_offset += 150
            
        elif msg_type == "ai":
            ai_content = msg.get("content", "")
            tool_calls = msg.get("tool_calls", [])
            
            nodes.append({
                "id": f"node_{node_id_counter}",
                "type": "agent",
                "data": {
                    "label": "Agent Decision",
                    "content": ai_content[:300] if ai_content else "(Tool calls only)",
                    "tool_calls": [
                        {
                            "name": tc.get("name"),
                            "args": str(tc.get("args", {}))[:100]
                        }
                        for tc in tool_calls
                    ],
                    "tool_count": len(tool_calls)
                },
                "position": {"x": 250, "y": y_offset}
            })
            edges.append({
                "id": f"edge_{len(edges)}",
                "source": last_node_id,
                "target": f"node_{node_id_counter}",
                "type": "smoothstep",
                "animated": True
            })
            last_node_id = f"node_{node_id_counter}"
            node_id_counter += 1
            y_offset += 200
            
            # Create tool execution nodes
            for tc in tool_calls:
                tool_node_id = f"node_{node_id_counter}"
                nodes.append({
                    "id": tool_node_id,
                    "type": "tool",
                    "data": {
                        "label": f"Tool: {tc.get('name')}",
                        "tool_name": tc.get("name"),
                        "args": tc.get("args", {}),
                        "call_id": tc.get("id"),
                    },
                    "position": {"x": 50 + (node_id_counter % 3) * 200, "y": y_offset}
                })
                edges.append({
                    "id": f"edge_{len(edges)}",
                    "source": last_node_id,
                    "target": tool_node_id,
                    "type": "smoothstep"
                })
                node_id_counter += 1
            
            if tool_calls:
                y_offset += 150
            
        elif msg_type == "tool":
            # Find the corresponding tool call node
            tool_call_id = msg.get("tool_call_id")
            tool_node = next((n for n in nodes if n.get("data", {}).get("call_id") == tool_call_id), None)
            
            if tool_node:
                # Update tool node with result
                result_content = msg.get("content", "")
                tool_node["data"]["result"] = result_content[:300]
                tool_node["data"]["success"] = "Error" not in result_content and "Failed" not in result_content
    
    # Add final output node
    response = doc.get("response", {})
    nodes.append({
        "id": f"node_{node_id_counter}",
        "type": "output",
        "data": {
            "label": "Final Output",
            "answer": response.get("answer"),
            "status": doc.get("status"),
            "execution_time_ms": doc.get("execution_time_ms"),
            "plan_steps": response.get("plan_steps"),
            "current_step": response.get("current_step"),
        },
        "position": {"x": 250, "y": y_offset}
    })
    edges.append({
        "id": f"edge_{len(edges)}",
        "source": last_node_id,
        "target": f"node_{node_id_counter}",
        "type": "smoothstep"
    })
    
    return {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "trace_id": str(doc["_id"]),
            "thread_id": doc.get("thread_id"),
            "workspace_id": doc.get("workspace_id"),
            "timestamp": doc["timestamp"].isoformat() if isinstance(doc.get("timestamp"), datetime) else doc.get("timestamp"),
            "execution_time_ms": doc.get("execution_time_ms"),
            "status": doc.get("status"),
            "total_nodes": len(nodes),
            "total_edges": len(edges),
        }
    }


@router.get("/dashboard/index.html", response_class=HTMLResponse)
async def get_dashboard():
    """Serve the interactive dashboard HTML."""
    html_path = os.path.join(os.path.dirname(__file__), "..", "..", "dashboard", "index.html")
    
    if os.path.exists(html_path):
        with open(html_path, "r") as f:
            return f.read()
    
    # Return inline dashboard if file doesn't exist
    return INLINE_DASHBOARD_HTML


# ── Inline Dashboard HTML ──────────────────────────────────────────
INLINE_DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Forge Agent Execution Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/react@18.2.0/umd/react.production.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/react-dom@18.2.0/umd/react-dom.production.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@babel/standalone@7.23.5/babel.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/reactflow@11.10.1/dist/umd/index.min.js"></script>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reactflow@11.10.1/dist/style.css">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0a0a0a; color: #fff; }
        .container { display: flex; height: 100vh; }
        .sidebar { width: 350px; background: #1a1a1a; border-right: 1px solid #333; overflow-y: auto; }
        .main { flex: 1; display: flex; flex-direction: column; }
        .header { background: #1a1a1a; padding: 20px; border-bottom: 1px solid #333; }
        .header h1 { font-size: 24px; font-weight: 600; }
        .flow-container { flex: 1; position: relative; }
        .trace-item { padding: 16px; border-bottom: 1px solid #333; cursor: pointer; transition: background 0.2s; }
        .trace-item:hover { background: #252525; }
        .trace-item.active { background: #2a2a2a; border-left: 3px solid #3b82f6; }
        .trace-header { display: flex; justify-content: space-between; align-items: start; margin-bottom: 8px; }
        .trace-id { font-size: 12px; font-family: monospace; color: #888; }
        .trace-time { font-size: 11px; color: #666; }
        .trace-info { font-size: 13px; color: #aaa; margin-top: 4px; }
        .status { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 500; }
        .status.success { background: #10b981; color: #fff; }
        .status.error { background: #ef4444; color: #fff; }
        .loading { padding: 40px; text-align: center; color: #666; }
        .node-custom { background: #1e293b; border: 2px solid #334155; border-radius: 8px; padding: 16px; min-width: 250px; }
        .node-input { border-color: #3b82f6; }
        .node-agent { border-color: #8b5cf6; }
        .node-tool { border-color: #10b981; }
        .node-output { border-color: #f59e0b; }
        .node-header { font-weight: 600; margin-bottom: 8px; font-size: 14px; }
        .node-content { font-size: 12px; color: #94a3b8; line-height: 1.5; }
        .tool-call { background: #334155; padding: 6px 10px; border-radius: 4px; margin: 4px 0; font-size: 11px; font-family: monospace; }
        .empty-state { padding: 60px 20px; text-align: center; color: #666; }
    </style>
</head>
<body>
    <div id="root"></div>
    
    <script type="text/babel">
        const { useState, useEffect, useCallback } = React;
        const { ReactFlow, Background, Controls, MiniMap } = ReactFlowRenderer;
        
        // Custom Node Components
        const InputNode = ({ data }) => (
            <div className="node-custom node-input">
                <div className="node-header">📥 {data.label}</div>
                <div className="node-content">
                    {data.question && <div><strong>Q:</strong> {data.question}</div>}
                    {data.content && <div>{data.content}</div>}
                    {data.attached_files > 0 && <div>📎 {data.attached_files} files</div>}
                </div>
            </div>
        );
        
        const AgentNode = ({ data }) => (
            <div className="node-custom node-agent">
                <div className="node-header">🤖 {data.label}</div>
                <div className="node-content">
                    {data.content && <div style={{marginBottom: '8px'}}>{data.content}</div>}
                    {data.tool_calls && data.tool_calls.length > 0 && (
                        <div>
                            <strong>Tools ({data.tool_count}):</strong>
                            {data.tool_calls.slice(0, 3).map((tc, i) => (
                                <div key={i} className="tool-call">{tc.name}</div>
                            ))}
                        </div>
                    )}
                </div>
            </div>
        );
        
        const ToolNode = ({ data }) => (
            <div className="node-custom node-tool">
                <div className="node-header">⚙️ {data.label}</div>
                <div className="node-content">
                    {data.result ? (
                        <div style={{color: data.success ? '#10b981' : '#ef4444'}}>
                            {data.result}
                        </div>
                    ) : (
                        <div style={{color: '#666'}}>Executing...</div>
                    )}
                </div>
            </div>
        );
        
        const OutputNode = ({ data }) => (
            <div className="node-custom node-output">
                <div className="node-header">✅ {data.label}</div>
                <div className="node-content">
                    {data.answer && <div><strong>Answer:</strong> {data.answer}</div>}
                    <div><strong>Time:</strong> {data.execution_time_ms}ms</div>
                    <div><strong>Status:</strong> {data.status}</div>
                    {data.plan_steps && (
                        <div style={{marginTop: '8px'}}>
                            <strong>Plan:</strong> {data.current_step}/{data.plan_steps.length} steps
                        </div>
                    )}
                </div>
            </div>
        );
        
        const nodeTypes = {
            input: InputNode,
            agent: AgentNode,
            tool: ToolNode,
            output: OutputNode,
        };
        
        // Main Dashboard Component
        function Dashboard() {
            const [traces, setTraces] = useState([]);
            const [selectedTrace, setSelectedTrace] = useState(null);
            const [flowData, setFlowData] = useState(null);
            const [loading, setLoading] = useState(true);
            
            useEffect(() => {
                fetchTraces();
            }, []);
            
            const fetchTraces = async () => {
                try {
                    const res = await fetch('/traces');
                    const data = await res.json();
                    setTraces(data.traces || []);
                    setLoading(false);
                } catch (err) {
                    console.error('Failed to fetch traces:', err);
                    setLoading(false);
                }
            };
            
            const loadTrace = async (traceId) => {
                try {
                    const res = await fetch(`/traces/${traceId}/flow`);
                    const data = await res.json();
                    setFlowData(data);
                    setSelectedTrace(traceId);
                } catch (err) {
                    console.error('Failed to load trace:', err);
                }
            };
            
            return (
                <div className="container">
                    <div className="sidebar">
                        <div style={{padding: '20px', borderBottom: '1px solid #333'}}>
                            <h2 style={{fontSize: '18px', fontWeight: 600}}>Execution Runs</h2>
                            <div style={{fontSize: '13px', color: '#666', marginTop: '4px'}}>
                                {traces.length} traces
                            </div>
                        </div>
                        {loading ? (
                            <div className="loading">Loading traces...</div>
                        ) : traces.length === 0 ? (
                            <div className="empty-state">No traces found</div>
                        ) : (
                            traces.map(trace => (
                                <div
                                    key={trace._id}
                                    className={`trace-item ${selectedTrace === trace._id ? 'active' : ''}`}
                                    onClick={() => loadTrace(trace._id)}
                                >
                                    <div className="trace-header">
                                        <div className="trace-id">{trace.thread_id?.slice(-12) || 'N/A'}</div>
                                        <span className={`status ${trace.status || 'success'}`}>
                                            {trace.status || 'success'}
                                        </span>
                                    </div>
                                    <div className="trace-info">
                                        ⏱ {trace.execution_time_ms}ms • 
                                        💬 {trace.message_count} msgs • 
                                        🔧 {trace.tool_call_count} tools
                                    </div>
                                    <div className="trace-time">
                                        {new Date(trace.timestamp).toLocaleString()}
                                    </div>
                                    {trace.has_plan && (
                                        <div style={{marginTop: '4px', fontSize: '11px', color: '#10b981'}}>
                                            📋 Has Plan
                                        </div>
                                    )}
                                </div>
                            ))
                        )}
                    </div>
                    
                    <div className="main">
                        <div className="header">
                            <h1>🔍 Agent Execution Flow</h1>
                            {flowData && (
                                <div style={{fontSize: '13px', color: '#888', marginTop: '8px'}}>
                                    {flowData.metadata.workspace_id} • {flowData.metadata.total_nodes} nodes • {flowData.metadata.total_edges} edges
                                </div>
                            )}
                        </div>
                        
                        <div className="flow-container">
                            {!flowData ? (
                                <div className="empty-state" style={{paddingTop: '100px'}}>
                                    Select a trace to visualize execution flow
                                </div>
                            ) : (
                                <ReactFlow
                                    nodes={flowData.nodes}
                                    edges={flowData.edges}
                                    nodeTypes={nodeTypes}
                                    fitView
                                    style={{background: '#0a0a0a'}}
                                >
                                    <Background color="#333" gap={16} />
                                    <Controls />
                                    <MiniMap />
                                </ReactFlow>
                            )}
                        </div>
                    </div>
                </div>
            );
        }
        
        ReactDOM.render(<Dashboard />, document.getElementById('root'));
    </script>
</body>
</html>
"""
