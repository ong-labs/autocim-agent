"""[checklist item 1] hitl_human_approval must be the ONLY pause mechanism.

`workflow.compile()` in graph.py must never register static
interrupt_before/interrupt_after nodes; pausing happens exclusively via the
dynamic `interrupt()` call inside hitl_node, and resuming happens exclusively
via `Command(resume=...)` -- never `graph.update_state()` (CLAUDE.md 5.B).
"""

import inspect

from langgraph.types import Command

import graph as graph_module
from graph import build_graph
from nodes.evaluator import MAX_RETRY_LIMIT


def test_compile_call_passes_no_interrupt_before_or_after():
    """Source-level guardrail directly on the workflow.compile(...) call."""
    source = inspect.getsource(graph_module.build_graph)
    compile_call = source[source.index("workflow.compile("):]
    assert "interrupt_before" not in compile_call
    assert "interrupt_after" not in compile_call


def test_compiled_graph_has_no_static_interrupts_registered():
    """Runtime guardrail: the compiled graph itself was never configured
    with static interrupt_before/interrupt_after node lists."""
    compiled = build_graph()
    assert compiled.interrupt_before_nodes == []
    assert compiled.interrupt_after_nodes == []


def test_dynamic_interrupt_pauses_at_hitl_and_resumes_with_command(state_factory, registered_bad_hw_config):
    """Full graph integration test: a never-converging HWConfig must drive
    the graph to pause at hitl_human_approval via dynamic interrupt() after
    MAX_RETRY_LIMIT failed attempts, and Command(resume=...) must safely
    continue execution with the sanitization contract (checklist item 3)
    applied on the way back through @planner.
    """
    compiled = build_graph()
    config = {"configurable": {"thread_id": "test-hitl-thread"}}
    init_state = state_factory(hw_spec_id=registered_bad_hw_config.hw_spec_id)

    result = compiled.invoke(init_state, config=config)

    assert "__interrupt__" in result, "graph must pause via dynamic interrupt(), not run to completion"
    assert result["needs_hitl"] is True
    assert result["retry_count"] == MAX_RETRY_LIMIT
    assert result["is_converged"] is False

    interrupt_obj = result["__interrupt__"][0]
    assert "reason" in interrupt_obj.value
    assert interrupt_obj.value["retry_count"] == MAX_RETRY_LIMIT

    # Resume and inspect the very next @planner update -- this is where the
    # sanitization contract (checklist item 3) must fire, right after
    # hitl_node's interrupt() return value is consumed. The bad HW spec
    # never converges, so a plain .invoke() here would keep looping through
    # another full retry cycle and land back on a *second* interrupt rather
    # than returning right after @planner -- so we inspect the update
    # stream instead of only the final return value.
    resumed_chunks = list(
        compiled.stream(
            Command(resume={"new_bounds": {"note": "researcher intervened"}}),
            config=config,
            stream_mode="updates",
        )
    )
    first_planner_update = next(chunk["planner"] for chunk in resumed_chunks if "planner" in chunk)

    assert first_planner_update["human_overrides"] == {}
    assert first_planner_update["needs_hitl"] is False
    assert first_planner_update["retry_count"] == 0
    assert first_planner_update["iteration_count"] == MAX_RETRY_LIMIT + 1

    # Still-bad HW correctly drives the graph through another full retry
    # cycle and back to a second dynamic interrupt, rather than looping
    # forever or silently giving up (regression coverage for the
    # unconditional-retry_count-reset bug fixed in nodes/planner.py).
    assert any("__interrupt__" in chunk for chunk in resumed_chunks)


def test_good_hw_config_converges_without_ever_interrupting(state_factory, registered_hw_config):
    """Control case: a converging HWConfig must reach END without ever
    pausing, proving interrupts are conditional (needs_hitl-driven), not
    unconditional/static.
    """
    compiled = build_graph()
    config = {"configurable": {"thread_id": "test-converge-thread"}}
    init_state = state_factory(hw_spec_id=registered_hw_config.hw_spec_id)

    result = compiled.invoke(init_state, config=config)

    assert "__interrupt__" not in result
    assert result["is_converged"] is True
