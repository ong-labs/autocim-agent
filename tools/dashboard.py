"""Static HTML dashboard for one AutoCIM-Agent run (state.py's `AutoCIMState`).

Real multi-researcher use needs more than `state["messages"]`'s free text
to build trust in / debug a run: which candidate was tried each iteration
and *why* (`planner_decisions`, appended by `nodes/planner.py`), what it
actually measured (`candidate_history`, appended by `nodes/evaluator.py`),
and how much the LLM cost along the way (`llm_usage`). This module joins
those three already-structured `AutoCIMState` fields -- the authoritative
numeric record -- into one self-contained HTML report.

`observability.py`'s JSON-Lines log is a separate, narrative/audit channel
(when/in-what-order things happened) -- this module reads `AutoCIMState`
directly, not that log, since state is the source of truth for the actual
numbers.

No charting dependency: the Pareto-front-movement chart is hand-rolled
inline SVG (a handful of <circle>/<polyline> elements), consistent with
this project's "no new heavy dependency" stance elsewhere (tools/search.py).
"""

from __future__ import annotations

import html
from typing import Any, Dict, List

from state import AutoCIMState


def _iteration_rows(state: AutoCIMState) -> List[Dict[str, Any]]:
    """One row per evaluated candidate, joining `candidate_history`
    (what happened) against `planner_decisions` (why it was tried) by
    `iteration` -- the two lists are appended by different nodes
    (nodes/evaluator.py, nodes/planner.py) so neither alone has both."""
    decisions = {d["iteration"]: d for d in state.get("planner_decisions", [])}
    rows = []
    for candidate in state.get("candidate_history", []):
        iteration = candidate.get("iteration")
        decision = decisions.get(iteration, {})
        rows.append(
            {
                **candidate,
                "search_tag": decision.get("search_tag"),
                "rationale": decision.get("rationale"),
                "used_llm": decision.get("used_llm"),
            }
        )
    return rows


def _svg_pareto_chart(rows: List[Dict[str, Any]], width: int = 480, height: int = 320, pad: int = 40) -> str:
    """accuracy (x) vs energy_pj (y, inverted so "better" trends upward),
    one point per iteration, connected in iteration order so the Pareto
    front's movement over the run is visible at a glance -- points still on
    the current Pareto front (pareto_rank == 1) are highlighted."""
    plottable = [r for r in rows if r.get("accuracy") is not None and r.get("energy_pj") is not None]
    if not plottable:
        return "<p>No evaluated candidates yet.</p>"

    accuracies = [r["accuracy"] for r in plottable]
    energies = [r["energy_pj"] for r in plottable]
    acc_min, acc_max = min(accuracies), max(accuracies)
    e_min, e_max = min(energies), max(energies)

    def x(acc: float) -> float:
        if acc_max == acc_min:
            return width / 2
        return pad + (acc - acc_min) / (acc_max - acc_min) * (width - 2 * pad)

    def y(energy: float) -> float:  # lower energy is better -> plotted higher
        if e_max == e_min:
            return height / 2
        return height - pad - (energy - e_min) / (e_max - e_min) * (height - 2 * pad)

    ordered = sorted(plottable, key=lambda r: r["iteration"])
    path = " ".join(f"{x(r['accuracy']):.1f},{y(r['energy_pj']):.1f}" for r in ordered)

    marks = []
    for r in ordered:
        cx, cy = x(r["accuracy"]), y(r["energy_pj"])
        color = "#2563eb" if r.get("pareto_rank") == 1 else "#9ca3af"
        tooltip = (
            f"iter {r['iteration']}: accuracy={r['accuracy']:.3f}, energy={r['energy_pj']:.2f}pJ, "
            f"rank={r.get('pareto_rank')}"
        )
        marks.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" fill="{color}" opacity="0.85">'
            f"<title>{html.escape(tooltip)}</title></circle>"
            f'<text x="{cx + 8:.1f}" y="{cy + 4:.1f}" font-size="10" fill="#374151">{r["iteration"]}</text>'
        )

    return (
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" '
        f'aria-label="accuracy vs energy per iteration; blue marks the current Pareto rank 1 candidates">'
        f'<polyline points="{path}" fill="none" stroke="#d1d5db" stroke-width="1"/>'
        + "".join(marks)
        + f'<text x="{pad}" y="{height - 8}" font-size="11" fill="#111827">accuracy &#8594;</text>'
        + f'<text x="8" y="{pad}" font-size="11" fill="#111827">energy_pj &#8593; (lower=better)</text>'
        + "</svg>"
    )


def _format_or_dash(value: Any, fmt: str = "{:.4f}") -> str:
    return fmt.format(value) if isinstance(value, (int, float)) else "-"


def _table_rows_html(rows: List[Dict[str, Any]]) -> str:
    lines = []
    for r in sorted(rows, key=lambda r: r["iteration"]):
        rationale = html.escape(str(r.get("rationale") or ""))
        lines.append(
            "<tr>"
            f"<td>{r['iteration']}</td>"
            f"<td>{html.escape(str(r.get('search_tag') or '-'))}</td>"
            f"<td>{_format_or_dash(r.get('accuracy'))}</td>"
            f"<td>{_format_or_dash(r.get('energy_pj'), '{:.2f}')}</td>"
            f"<td>{_format_or_dash(r.get('noc_latency_ms'), '{:.3f}')}</td>"
            f"<td>{r.get('pareto_rank', '-')}</td>"
            f"<td>{'yes' if r.get('is_converged') else 'no'}</td>"
            f'<td title="{rationale}">{rationale[:80]}{"..." if len(rationale) > 80 else ""}</td>'
            "</tr>"
        )
    return "\n".join(lines)


def _llm_usage_summary(state: AutoCIMState) -> Dict[str, Any]:
    usage = state.get("llm_usage", [])
    costs = [u["estimated_cost_usd"] for u in usage if u.get("estimated_cost_usd") is not None]
    return {
        "n_calls": len(usage),
        "total_attempts": sum(u.get("attempts", 0) for u in usage),
        "n_failed": sum(1 for u in usage if not u.get("succeeded")),
        "total_cost_usd": sum(costs) if costs else None,
    }


def render_dashboard_html(state: AutoCIMState) -> str:
    """Pure function of `state` -- no I/O, no LangGraph/checkpointer
    dependency, so it's directly unit-testable against a synthetic state
    (state_factory in tests) exactly like the rest of this project's
    Mock-First-tested tools."""
    rows = _iteration_rows(state)
    usage = _llm_usage_summary(state)
    cost_text = f"${usage['total_cost_usd']:.4f}" if usage["total_cost_usd"] is not None else "n/a (no AUTOCIM_LLM_COST_PER_1K_* configured)"

    title = f"{state.get('model_id')} / {state.get('hw_spec_id')}"

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>AutoCIM-Agent run: {html.escape(title)}</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #111827; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 0.5rem; }}
  th, td {{ border: 1px solid #e5e7eb; padding: 6px 10px; text-align: left; font-size: 13px; }}
  th {{ background: #f9fafb; }}
  .summary {{ color: #374151; }}
</style>
</head>
<body>
<h1>AutoCIM-Agent run</h1>
<p class="summary">
  model_id={html.escape(str(state.get("model_id")))}
  hw_spec_id={html.escape(str(state.get("hw_spec_id")))}
  iteration_count={state.get("iteration_count")}
  is_converged={state.get("is_converged")}
</p>
<p class="summary">
  LLM calls: {usage["n_calls"]} node-invocation(s), {usage["total_attempts"]} total attempt(s),
  {usage["n_failed"]} failed after retries, estimated cost: {cost_text}
</p>

<h2>Pareto front movement (accuracy vs energy, blue = current rank 1)</h2>
{_svg_pareto_chart(rows)}

<h2>Per-iteration candidates</h2>
<table>
<thead><tr><th>iter</th><th>search phase</th><th>accuracy</th><th>energy_pj</th><th>latency_ms</th><th>pareto_rank</th><th>converged</th><th>rationale</th></tr></thead>
<tbody>
{_table_rows_html(rows)}
</tbody>
</table>
</body>
</html>"""
