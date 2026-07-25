"""Real session persistence: a session paused at `hitl_human_approval` (or
interrupted mid-prompt) must survive the *process* exiting, not just
survive within one Python object's lifetime. These tests build the graph
against a real on-disk SQLite file via a fresh `SqliteSaver` instance for
each phase -- simulating a genuinely separate CLI invocation -- rather than
reusing the same checkpointer object throughout, which would only prove
in-memory reuse works (already true of the old MemorySaver) and say nothing
about surviving a restart.
"""

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

import main
from graph import build_graph
from main import get_pending_interrupt, run_session
from nodes.evaluator import MAX_RETRY_LIMIT


def test_pending_interrupt_survives_a_fresh_checkpointer_on_the_same_file(
    tmp_path, state_factory, registered_bad_hw_config
):
    db_path = str(tmp_path / "checkpoints.sqlite")
    config = {"configurable": {"thread_id": "persist-thread-1"}}
    init_state = state_factory(hw_spec_id=registered_bad_hw_config.hw_spec_id)

    with SqliteSaver.from_conn_string(db_path) as checkpointer_a:
        graph_a = build_graph(checkpointer=checkpointer_a)
        graph_a.invoke(init_state, config=config)  # bad HW -> pauses at hitl after MAX_RETRY_LIMIT
        assert get_pending_interrupt(graph_a, config) is not None

    # A brand-new SqliteSaver + compiled graph over the *same file* --
    # nothing here is shared with graph_a except the on-disk database.
    with SqliteSaver.from_conn_string(db_path) as checkpointer_b:
        graph_b = build_graph(checkpointer=checkpointer_b)
        interrupt_payload = get_pending_interrupt(graph_b, config)

        assert interrupt_payload is not None
        assert interrupt_payload["retry_count"] == MAX_RETRY_LIMIT

        # bad HW never converges, so a plain .invoke() would run all the way
        # to a *second* interrupt -- inspect the update stream instead, same
        # technique as test_e2e_hitl.py, to check the very next @planner
        # update (where sanitization fires) rather than the final state.
        resumed_chunks = list(
            graph_b.stream(
                Command(resume={"new_bounds": {"note": "resumed in a new process"}}),
                config=config,
                stream_mode="updates",
            )
        )
        first_planner_update = next(chunk["planner"] for chunk in resumed_chunks if "planner" in chunk)
        assert first_planner_update["human_overrides"] == {}  # sanitization contract still holds across the restart
        assert first_planner_update["retry_count"] == 0


def test_run_session_resumes_persisted_pause_after_eof_then_converges(
    monkeypatch, tmp_path, registered_bad_hw_config, good_hw_config
):
    db_path = str(tmp_path / "checkpoints.sqlite")
    thread_id = "persist-thread-2"

    # Phase 1: researcher's terminal closes (EOFError) right at the HITL
    # prompt -- run_session must persist the pause and return cleanly,
    # not lose the session or raise.
    monkeypatch.setattr(main, "prompt_for_override", lambda payload: (_ for _ in ()).throw(EOFError()))
    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        run_session("resnet18", registered_bad_hw_config, thread_id, checkpointer)

    # Phase 2: a new process (fresh SqliteSaver on the same file) re-invokes
    # with the same --thread-id. Swap in a HWConfig that actually converges,
    # registered under the *same* hw_spec_id the paused session already
    # carries in its state (run_session's own register_hw_config call does
    # this) -- run_session must detect the persisted pause and resume it,
    # not silently start over from a fresh initial state.
    resumed_hw_config = good_hw_config.model_copy(update={"hw_spec_id": registered_bad_hw_config.hw_spec_id})
    monkeypatch.setattr(main, "prompt_for_override", lambda payload: {"new_bounds": {}})
    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        run_session("resnet18", resumed_hw_config, thread_id, checkpointer)
        final_state = build_graph(checkpointer=checkpointer).get_state({"configurable": {"thread_id": thread_id}}).values

    assert final_state["is_converged"] is True
    assert final_state["iteration_count"] > MAX_RETRY_LIMIT  # ran the interrupted iteration plus at least one more


def test_run_session_does_not_rerun_an_already_completed_thread(monkeypatch, tmp_path, registered_hw_config):
    db_path = str(tmp_path / "checkpoints.sqlite")
    thread_id = "persist-thread-3"

    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        run_session("resnet18", registered_hw_config, thread_id, checkpointer)  # good HW -> converges, no HITL

    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("stream_until_interrupt must not run again for an already-completed thread")

    monkeypatch.setattr(main, "stream_until_interrupt", _must_not_be_called)
    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        run_session("resnet18", registered_hw_config, thread_id, checkpointer)  # must short-circuit, not re-invoke
