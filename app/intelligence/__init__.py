"""
Intelligence System - All 3 phases of agent intelligence.

Phase 1: Persistent Memory (workspace_memory)
Phase 2: LLM-Powered Intelligence (error_analyzer, adaptive_config)
Phase 3: Long-term Intelligence (hierarchical_planner, checkpoints, model_router, parallel)
"""

# Phase 1
from .phase1.workspace_memory import (
    load_workspace_memory,
    save_workspace_memory,
    record_failure,
    record_success,
    is_exhausted_approach,
    get_failure_summary,
    should_ask_for_help,
    clear_workspace_memory,
)

# Phase 2
from .phase2.error_analyzer import (
    analyze_error_with_llm,
    compare_commands_semantically,
    analyze_user_intent,
)

from .phase2.adaptive_config import (
    load_adaptive_config,
    update_threshold_based_on_outcome,
    record_outcome_for_learning,
    get_config_summary,
)

# Phase 3
from .phase3.hierarchical_planner import (
    HierarchicalPlan,
    detect_if_should_break_into_subtasks,
    suggest_alternative_approach,
    visualize_plan_tree,
)

from .phase3.learning_checkpoints import (
    create_checkpoint_with_llm,
    format_checkpoint_for_prompt,
    should_create_checkpoint,
    merge_checkpoints,
)

from .phase3.model_router import (
    get_optimal_model_for_turn,
    analyze_task_requirements,
    select_model_based_on_analysis,
)

from .phase3.parallel_executor import (
    execute_with_optimal_parallelism,
    can_run_in_parallel,
    group_tool_calls_by_independence,
)

__all__ = [
    # Phase 1
    "load_workspace_memory",
    "save_workspace_memory",
    "record_failure",
    "record_success",
    "is_exhausted_approach",
    "get_failure_summary",
    "should_ask_for_help",
    "clear_workspace_memory",
    # Phase 2
    "analyze_error_with_llm",
    "compare_commands_semantically",
    "analyze_user_intent",
    "load_adaptive_config",
    "update_threshold_based_on_outcome",
    "record_outcome_for_learning",
    "get_config_summary",
    # Phase 3
    "HierarchicalPlan",
    "detect_if_should_break_into_subtasks",
    "suggest_alternative_approach",
    "visualize_plan_tree",
    "create_checkpoint_with_llm",
    "format_checkpoint_for_prompt",
    "should_create_checkpoint",
    "merge_checkpoints",
    "get_optimal_model_for_turn",
    "analyze_task_requirements",
    "select_model_based_on_analysis",
    "execute_with_optimal_parallelism",
    "can_run_in_parallel",
    "group_tool_calls_by_independence",
]
