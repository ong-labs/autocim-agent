"""AutoCIM-Agent LangGraph assembly (Research_Plan.md 2절).

START -> planner -> tuner -> (mapper, profiler fan-out) -> verifier -> evaluator
                                                                            |
                            +----------------- Re-plan with History -------+
                            |                                              |
                       hitl_human_approval <--- HITL Interrupt ------------+
                            |                                              |
                            +----------------------> planner               |
                                                                            +--> Converged (Done) -> precision_verifier
                                                                                                            |
                                                    +-------------- Re-plan with History -------------------+
                                                    |                                                       |
                                               hitl_human_approval <--- HITL Interrupt ---------------------+
                                                    |                                                       |
                                                    +-> planner                                             +--> Converged (Done) -> END

@precision_verifier (mock Stage-2 high-fidelity re-check, CLAUDE.md 5.D Mock
First) sits between @evaluator's fast-approximation convergence call and a
real `END` -- @evaluator's own "Converged (Done)" routes here first instead
of straight to `END`, and this node re-routes through the exact same three
outcomes @evaluator_router already defines (reused directly, not
duplicated) once its own precision check confirms (or overturns) that
verdict. Only a candidate the fast approximation already thinks is done
ever reaches this node -- never every candidate a search tries.

HITL pausing uses only the dynamic `interrupt()` call inside `hitl_node`
(checklist item 1 / CLAUDE.md 5.B) -- `workflow.compile()` below must never
be given `interrupt_before`/`interrupt_after`.
"""

from typing import Optional

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from nodes import (
    evaluator_node,
    evaluator_router,
    hitl_node,
    mapper_node,
    planner_node,
    precision_verifier_node,
    profiler_node,
    tuner_node,
    verifier_node,
)
from state import AutoCIMState


def build_graph(checkpointer: Optional[BaseCheckpointSaver] = None) -> CompiledStateGraph:
    workflow = StateGraph(AutoCIMState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("tuner", tuner_node)
    workflow.add_node("mapper", mapper_node)
    workflow.add_node("profiler", profiler_node)
    workflow.add_node("verifier", verifier_node)
    workflow.add_node("evaluator", evaluator_node)
    workflow.add_node("precision_verifier", precision_verifier_node)
    workflow.add_node("hitl_human_approval", hitl_node)

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "tuner")

    # Fan-out: mapper/profiler run in parallel, then fan back into verifier.
    workflow.add_edge("tuner", "mapper")
    workflow.add_edge("tuner", "profiler")
    workflow.add_edge("mapper", "verifier")
    workflow.add_edge("profiler", "verifier")

    workflow.add_edge("verifier", "evaluator")
    workflow.add_edge("hitl_human_approval", "planner")

    workflow.add_conditional_edges(
        "evaluator",
        evaluator_router,
        {
            # Was END directly before @precision_verifier existed -- the
            # fast-approximation's own "converged" verdict is no longer
            # final by itself, only a trigger for the (rare, expensive)
            # precise re-check.
            "Converged (Done)": "precision_verifier",
            "HITL Interrupt": "hitl_human_approval",
            "Re-plan with History": "planner",
        },
    )
    workflow.add_conditional_edges(
        "precision_verifier",
        evaluator_router,  # same three outcomes, same state fields -- see module docstring
        {
            "Converged (Done)": END,
            "HITL Interrupt": "hitl_human_approval",
            "Re-plan with History": "planner",
        },
    )

    # Checklist item 1 (strict guardrail): dynamic interrupt() inside
    # hitl_node is the only pause mechanism -- compile() must not declare
    # interrupt_before/interrupt_after.
    #
    # Defaults to MemorySaver (in-process only -- a session dies with the
    # process, exactly the limitation main.py's run_session works around
    # for real CLI sessions) so every existing caller (tests, the module-
    # level `graph` singleton below) keeps today's fully-isolated-per-call
    # behavior with zero changes. main.py passes a real persistent
    # checkpointer (SqliteSaver) for actual CLI sessions.
    return workflow.compile(checkpointer=checkpointer or MemorySaver())


graph = build_graph()
