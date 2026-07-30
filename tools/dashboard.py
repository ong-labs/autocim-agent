"""Static HTML dashboard for one AutoCIM-Agent run (state.py's `AutoCIMState`).

Real multi-researcher use needs more than `state["messages"]`'s free text
to build trust in / debug a run: which candidate was tried each iteration
and *why* (`planner_decisions`, appended by `nodes/planner.py`), what it
actually measured (`candidate_history`, appended by `nodes/evaluator.py`),
and how many LLM calls it took, including retries/failures (`llm_usage`).
This module joins those three already-structured `AutoCIMState` fields -- the authoritative
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
import re
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
                "rationale_ko": decision.get("rationale_ko"),
                "used_llm": decision.get("used_llm"),
            }
        )
    return rows


_WARMUP_TAG_RE = re.compile(r"^LHS warm-up candidate (\d+)/(\d+)$")
_SURROGATE_TAG_RE = re.compile(r"^surrogate-model acquisition over (\d+) prior candidates$")


def _bi(en: str, ko: str) -> str:
    """Wraps static UI prose (never LLM/data-derived text -- rationale,
    search_tag, model_id/hw_spec_id stay in their original language,
    see render_dashboard_html) in both languages; the JS toggle in
    render_dashboard_html's <script> flips which one has the `hidden`
    attribute. Both are always in the DOM so switching language doesn't
    reflow/lengthen the page -- only the previously-hidden span un-hides
    in place of the one that just hid."""
    return f'<span data-lang="en">{en}</span><span data-lang="ko" hidden>{ko}</span>'


def _search_phase_summary_html(rows: List[Dict[str, Any]]) -> str:
    """Explains *why* the per-iteration table below can show the same
    "LHS warm-up candidate" search phase many times in a row: warm-up is a
    fixed-size, uniform space-filling sample (`nodes/planner.py`'s
    `warmup_count`) whose total is baked into its own tag text ("X/N"),
    parsed back out here rather than re-derived from model/hw config (this
    module intentionally has no dependency on that, see module docstring)
    -- so a reader watching a run doesn't have to infer "when does this
    stop" from a wall of repeated rows alone. Deliberately does not invent
    a total for the optimization phase that follows warm-up: it really has
    none (runs until a convergence target is hit or the run is stopped)."""
    ordered = sorted(rows, key=lambda r: r["iteration"]) if rows else []
    if not ordered:
        return f'<p class="phase-note">{_bi("No candidates evaluated yet.", "아직 평가된 후보가 없습니다.")}</p>'

    latest = ordered[-1]
    tag = str(latest.get("search_tag") or "")
    warmup_match = _WARMUP_TAG_RE.match(tag)
    surrogate_match = _SURROGATE_TAG_RE.match(tag)

    if warmup_match:
        done, total = int(warmup_match.group(1)), int(warmup_match.group(2))
        pct = min(100, round(done / total * 100)) if total else 0
        title_en = f"Warm-up phase &mdash; {done}/{total} candidates ({pct}%)"
        title_ko = f"워밍업(warm-up) 단계 &mdash; {done}/{total} 완료 ({pct}%)"
        note_en = (
            "Warm-up uniformly samples the search space and always runs exactly "
            f"<strong>{total}</strong> times for this model/hardware, regardless of how it's going &mdash; "
            "that count is fixed up front, not a sign of trouble. Once it finishes, the optimizer switches "
            "to surrogate-model-guided search, which has <strong>no fixed total</strong>: it keeps proposing "
            "candidates until a convergence target is met or the run is stopped."
        )
        note_ko = (
            "워밍업은 탐색 공간을 균일하게 샘플링하는 단계로, 진행 상황과 무관하게 이 모델/하드웨어 기준 "
            f"항상 정확히 <strong>{total}</strong>번 실행됩니다 &mdash; 미리 정해진 횟수일 뿐, 문제가 있다는 "
            "신호가 아닙니다. 이 단계가 끝나면 서로게이트 모델 기반 탐색으로 전환되며, 그 단계는 "
            "<strong>정해진 총 횟수가 없습니다</strong>: 수렴 목표를 만족하거나 실행이 중단될 때까지 후보 "
            "제안을 계속합니다."
        )
        return (
            '<div class="phase-box phase-warmup">'
            f'<div class="phase-title">{_bi(title_en, title_ko)}</div>'
            f'<div class="phase-bar"><div class="phase-bar-fill" style="width:{pct}%"></div></div>'
            f'<p class="phase-note">{_bi(note_en, note_ko)}</p>'
            "</div>"
        )
    if surrogate_match:
        n_prior = int(surrogate_match.group(1))
        title_en = f"Optimization phase &mdash; surrogate-model-guided search ({n_prior} prior candidate(s) so far)"
        title_ko = f"최적화 단계 &mdash; 서로게이트 모델 기반 탐색 (지금까지 이전 후보 {n_prior}개 반영)"
        note_en = (
            "Warm-up is complete. From here there is no fixed iteration count: the optimizer keeps proposing "
            "candidates until a convergence target is met (if one was set) or the run is stopped."
        )
        note_ko = (
            "워밍업이 완료되었습니다. 이제부터는 정해진 반복 횟수가 없습니다: 수렴 목표가 설정되어 있다면 "
            "이를 만족하거나, 실행이 중단될 때까지 최적화 탐색을 계속합니다."
        )
        return (
            '<div class="phase-box phase-search">'
            f'<div class="phase-title">{_bi(title_en, title_ko)}</div>'
            f'<p class="phase-note">{_bi(note_en, note_ko)}</p>'
            "</div>"
        )
    return ""


def _svg_pareto_chart(rows: List[Dict[str, Any]], width: int = 480, height: int = 320, pad: int = 40) -> str:
    """energy_pj (x) vs accuracy (y, inverted so higher accuracy trends
    upward), one point per iteration, connected in iteration order so the
    Pareto front's movement over the run is visible at a glance -- points
    still on the current Pareto front (pareto_rank == 1) are highlighted.

    Axis label text is anchored in the margin outside the
    [pad, width-pad] x [pad, height-pad] plot rectangle (the y-axis label
    rotated, centered on the left margin) rather than at a fixed (x, y)
    inside it -- the old top-left placement sat at exactly (pad, pad),
    which is also where the min-energy/max-accuracy point can land, so the
    label text and a data point could overlap."""
    plottable = [r for r in rows if r.get("accuracy") is not None and r.get("energy_pj") is not None]
    if not plottable:
        return "<p>No evaluated candidates yet.</p>"

    accuracies = [r["accuracy"] for r in plottable]
    energies = [r["energy_pj"] for r in plottable]
    acc_min, acc_max = min(accuracies), max(accuracies)
    e_min, e_max = min(energies), max(energies)

    def x(energy: float) -> float:  # energy_pj increases left -> right
        if e_max == e_min:
            return width / 2
        return pad + (energy - e_min) / (e_max - e_min) * (width - 2 * pad)

    def y(accuracy: float) -> float:  # higher accuracy is better -> plotted higher
        if acc_max == acc_min:
            return height / 2
        return height - pad - (accuracy - acc_min) / (acc_max - acc_min) * (height - 2 * pad)

    ordered = sorted(plottable, key=lambda r: r["iteration"])
    path = " ".join(f"{x(r['energy_pj']):.1f},{y(r['accuracy']):.1f}" for r in ordered)

    marks = []
    for r in ordered:
        cx, cy = x(r["energy_pj"]), y(r["accuracy"])
        color = "#5b9dff" if r.get("pareto_rank") == 1 else "#5c6b7e"
        tooltip = (
            f"iter {r['iteration']}: energy={r['energy_pj']:.2f}pJ, accuracy={r['accuracy']:.3f}, "
            f"rank={r.get('pareto_rank')}"
        )
        marks.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" fill="{color}" opacity="0.9">'
            f"<title>{html.escape(tooltip)}</title></circle>"
            f'<text x="{cx + 8:.1f}" y="{cy + 4:.1f}" font-size="10" fill="#c8d3e0">{r["iteration"]}</text>'
        )

    return (
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" '
        f'aria-label="energy vs accuracy per iteration; blue marks the current Pareto rank 1 candidates">'
        f'<polyline points="{path}" fill="none" stroke="#33475c" stroke-width="1"/>'
        + "".join(marks)
        + f'<text x="{width / 2:.1f}" y="{height - 8}" font-size="11" fill="#93a4b8" '
        f'text-anchor="middle">energy_pj &#8594; (lower=better)</text>'
        + f'<text x="14" y="{height / 2:.1f}" font-size="11" fill="#93a4b8" text-anchor="middle" '
        f'transform="rotate(-90 14 {height / 2:.1f})">accuracy &#8593;</text>'
        + "</svg>"
    )


def _pareto_legend_html() -> str:
    """Separate from `_svg_pareto_chart` since these are two plain HTML
    swatches, not SVG -- simpler to keep them (and their bilingual labels,
    `_bi`) out of the hand-rolled SVG string entirely."""
    return (
        '<div class="legend">'
        '<span class="legend-item"><span class="legend-swatch" style="background:#5b9dff"></span>'
        f'{_bi("Pareto rank 1 (on current front)", "파레토 랭크 1 (현재 프론트)")}</span>'
        '<span class="legend-item"><span class="legend-swatch" style="background:#5c6b7e"></span>'
        f'{_bi("Not on current front (dominated)", "현재 프론트에 속하지 않음 (열세 후보)")}</span>'
        "</div>"
    )


def _format_or_dash(value: Any, fmt: str = "{:.4f}") -> str:
    return fmt.format(value) if isinstance(value, (int, float)) else "-"


def _row_html(r: Dict[str, Any]) -> str:
    """Rationale is rendered in full (no char-count truncation) -- a
    hard `[:80]` slice cut mid-word/mid-sentence, which read as broken
    rather than intentionally short. `.rationale-cell`'s CSS wraps it
    into a fixed-width column instead, so the complete text is always
    visible without stretching the row or clipping anything.

    rationale_ko (schemas/planner.py's `PlannerLayerDecision.rationale_ko`,
    the @planner LLM's own Korean translation of the same justification --
    NOT machine-translated here) follows the same EN/KO toggle as the rest
    of the page's static UI text (`_bi`). Older recorded decisions have no
    rationale_ko (field didn't exist yet) -- those fall back to showing the
    English rationale in both language modes rather than an empty cell."""
    rationale = html.escape(str(r.get("rationale") or ""))
    rationale_ko = r.get("rationale_ko")
    rationale_html = _bi(rationale, html.escape(rationale_ko) if rationale_ko else rationale)
    return (
        "<tr>"
        f"<td>{r['iteration']}</td>"
        f"<td>{html.escape(str(r.get('search_tag') or '-'))}</td>"
        f"<td>{_format_or_dash(r.get('accuracy'))}</td>"
        f"<td>{_format_or_dash(r.get('energy_pj'), '{:.2f}')}</td>"
        f"<td>{_format_or_dash(r.get('noc_latency_ms'), '{:.3f}')}</td>"
        f"<td>{r.get('pareto_rank', '-')}</td>"
        f"<td>{_bi('yes', '예') if r.get('is_converged') else _bi('no', '아니요')}</td>"
        f'<td class="rationale-cell">{rationale_html}</td>'
        "</tr>"
    )


def _is_warmup_row(r: Dict[str, Any]) -> bool:
    return bool(_WARMUP_TAG_RE.match(str(r.get("search_tag") or "")))


def _warmup_group_html(group: List[Dict[str, Any]]) -> str:
    """Collapses a contiguous run of LHS warm-up rows into one
    <details>-backed summary row (native disclosure, no JS) instead of
    `warmup_count(n_stages)` individual ones -- at the real max
    (nodes/planner.py's `_MAX_WARMUP_CANDIDATES` = 12) those would
    otherwise dominate the table before the surrogate-guided search even
    starts, and (unlike a real planner decision) their rationale never
    varies with anything a reader would compare row-to-row (tools/
    batch_warmup.py skips the LLM entirely). Expands in place -- collapsed
    by default so the page isn't taller than it needs to be, matching this
    dashboard's EN/KO toggle's same "don't lengthen the page" approach."""
    first, last = group[0]["iteration"], group[-1]["iteration"]
    summary_en = f"Warm-up batch (iterations {first}-{last}, {len(group)} candidates, no LLM confirmation)"
    summary_ko = f"워밍업 배치 (반복 {first}-{last}, {len(group)}개 후보, LLM 확인 없음)"
    inner_rows = "\n".join(_row_html(r) for r in group)
    return (
        '<tr class="warmup-group-row">'
        '<td colspan="8" class="warmup-group-cell">'
        "<details class=\"warmup-details\">"
        f"<summary>{_bi(summary_en, summary_ko)}</summary>"
        '<table class="warmup-inner-table"><tbody>'
        f"{inner_rows}"
        "</tbody></table>"
        "</details>"
        "</td>"
        "</tr>"
    )


def _table_rows_html(rows: List[Dict[str, Any]]) -> str:
    ordered = sorted(rows, key=lambda r: r["iteration"])
    blocks: List[str] = []
    i = 0
    while i < len(ordered):
        if _is_warmup_row(ordered[i]):
            j = i + 1
            while j < len(ordered) and _is_warmup_row(ordered[j]):
                j += 1
            group = ordered[i:j]
            # A lone warm-up row isn't worth collapsing behind a click --
            # only a real multi-row batch is.
            blocks.append(_warmup_group_html(group) if len(group) >= 2 else _row_html(group[0]))
            i = j
        else:
            blocks.append(_row_html(ordered[i]))
            i += 1
    return "\n".join(blocks)


def _calibration_section_html(state: AutoCIMState) -> str:
    """Whether this run's energy/latency numbers are grounded in a real
    published reference (`tools.calibration`) or are still the uncorrected
    analytical fast-approximation (`tools.cim_physics`) -- the honest stand-
    in for "precise simulator vs fast-approximation error margin" this
    project can actually make, given @mapper/@profiler don't bind a real
    NeuroSim/CIM-Loop instance (CLAUDE.md 5.D Mock First). Uncalibrated is
    not an error state -- it's the expected case for any hw_spec_id that
    doesn't exactly match a known reference -- but it belongs on the
    dashboard, not buried in a code comment, since it directly bears on how
    much to trust the absolute accuracy/energy/latency numbers below."""
    hw_spec_id = state.get("hw_spec_id")
    factor = (state.get("calibration_factors") or {}).get(hw_spec_id)
    provenance = (state.get("calibration_provenance") or {}).get(hw_spec_id)
    hw_spec_id_esc = html.escape(str(hw_spec_id))

    if factor is None or provenance is None:
        warn_en = (
            f"&#9888; hw_spec_id={hw_spec_id_esc} is <strong>uncalibrated</strong> -- no published "
            "literature reference exactly matches this hardware configuration (tools/calibration.py). "
            "energy_pj/latency_ns figures below are analytical order-of-magnitude estimates "
            "(tools/cim_physics.py), not validated against precise simulation or silicon. "
            "Safe for comparing candidates <em>within this run</em>; do not treat the absolute numbers "
            "as a validated benchmark."
        )
        warn_ko = (
            f"&#9888; hw_spec_id={hw_spec_id_esc}는 <strong>캘리브레이션되지 않았습니다</strong> -- 이 하드웨어 "
            "구성과 정확히 일치하는 공개 문헌 참조값이 없습니다 (tools/calibration.py). 아래 energy_pj/"
            "latency_ns 수치는 정밀 시뮬레이션이나 실측 대비 검증되지 않은, 분석적인 order-of-magnitude "
            "추정값(tools/cim_physics.py)입니다. <em>이번 실행 내에서</em> 후보끼리 비교하는 용도로는 "
            "안전하지만, 절대값을 검증된 벤치마크로 취급하지 마십시오."
        )
        return f'<p class="calib-warning">{_bi(warn_en, warn_ko)}</p>'

    if provenance.get("precision_verified"):
        raw_energy = provenance.get("raw_energy_pj")
        precise_energy = provenance.get("precise_energy_pj")
        precision_ok_en = (
            f"&#10003; hw_spec_id={hw_spec_id_esc} is calibrated: analytical energy is scaled by "
            f"<strong>{factor:.4f}x</strong> against this run's own Stage-2 precision re-check "
            f"({raw_energy} &rarr; {precise_energy} pJ, iteration {provenance.get('iteration')}) -- "
            "<em>mock placeholder re-check (tools/precision_check.py), not a real NeuroSim/CIM-Loop "
            "measurement yet</em>."
        )
        precision_ok_ko = (
            f"&#10003; hw_spec_id={hw_spec_id_esc}는 캘리브레이션되었습니다: 분석적 에너지 추정값을 이번 실행 "
            f"자체의 2단계 정밀 재검증 결과({raw_energy} &rarr; {precise_energy} pJ, iteration "
            f"{provenance.get('iteration')}) 대비 <strong>{factor:.4f}배</strong>로 보정했습니다 -- <em>실제 "
            "NeuroSim/CIM-Loop 측정이 아닌 mock 자리 표시자 재검증(tools/precision_check.py)입니다</em>."
        )
        return f'<p class="calib-ok">{_bi(precision_ok_en, precision_ok_ko)}</p>'

    source = html.escape(str(provenance.get("source") or ""))
    note = html.escape(str(provenance.get("note") or ""))
    ref_value = provenance.get("reference_energy_pj_per_mac")
    ok_en = (
        f"&#10003; hw_spec_id={hw_spec_id_esc} is calibrated: analytical energy is scaled by "
        f"<strong>{factor:.4f}x</strong> against a real published reference ({ref_value} pJ/MAC) -- "
        f"<em>{source}</em>."
    )
    ok_ko = (
        f"&#10003; hw_spec_id={hw_spec_id_esc}는 캘리브레이션되었습니다: 분석적 에너지 추정값을 실제 공개된 "
        f"참조값({ref_value} pJ/MAC) 대비 <strong>{factor:.4f}배</strong>로 보정했습니다 -- <em>{source}</em>."
    )
    caveat_html = ""
    if note:
        caveat_html = "<br>" + _bi(
            f"<strong>Stated uncertainty:</strong> {note}", f"<strong>명시된 불확실성:</strong> {note}"
        )
    return f'<p class="calib-ok">{_bi(ok_en, ok_ko)}{caveat_html}</p>'


def _llm_usage_summary(state: AutoCIMState) -> Dict[str, Any]:
    usage = state.get("llm_usage", [])
    return {
        "n_calls": len(usage),
        "total_attempts": sum(u.get("attempts", 0) for u in usage),
        "n_failed": sum(1 for u in usage if not u.get("succeeded")),
    }


def render_dashboard_html(state: AutoCIMState) -> str:
    """Pure function of `state` -- no I/O, no LangGraph/checkpointer
    dependency, so it's directly unit-testable against a synthetic state
    (state_factory in tests) exactly like the rest of this project's
    Mock-First-tested tools."""
    rows = _iteration_rows(state)
    usage = _llm_usage_summary(state)

    title = f"{state.get('model_id')} / {state.get('hw_spec_id')}"
    converged = state.get("is_converged")
    llm_summary_en = (
        f'LLM calls: {usage["n_calls"]} node-invocation(s), {usage["total_attempts"]} total attempt(s), '
        f'{usage["n_failed"]} failed after retries'
    )
    llm_summary_ko = (
        f'LLM 호출: {usage["n_calls"]}회(노드 호출 기준), 총 시도 {usage["total_attempts"]}회, '
        f'재시도 후 실패 {usage["n_failed"]}건'
    )

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>AutoCIM-Agent report: {html.escape(title)}</title>
<style>
  :root {{
    color-scheme: dark;
    --bg: #0b0f14;
    --bg-panel: #111821;
    --bg-panel-alt: #161f2b;
    --border: #253242;
    --text: #e6edf3;
    --text-muted: #93a4b8;
    --blue: #5b9dff;
    --amber-bg: #241d0d;
    --amber-border: #4d3a14;
    --amber-text: #ecd9a8;
    --green-bg: #0f2019;
    --green-border: #1e4432;
    --green-text: #bfe9d3;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, "Segoe UI", system-ui, sans-serif;
    margin: 0;
    padding: 2.5rem clamp(1rem, 4vw, 3rem) 4rem;
    background: var(--bg);
    color: var(--text);
    line-height: 1.55;
  }}
  h1 {{ font-size: 1.35rem; font-weight: 600; margin: 0; }}
  h2 {{
    font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: .06em;
    color: var(--text-muted); border-bottom: 1px solid var(--border); padding-bottom: .5rem;
    margin: 2.4rem 0 1rem;
  }}
  .header-row {{ display: flex; align-items: center; justify-content: space-between; gap: 1rem; flex-wrap: wrap; }}
  .lang-toggle {{ display: flex; gap: 4px; }}
  .lang-btn {{
    font-family: ui-monospace, "SFMono-Regular", Consolas, monospace;
    font-size: 12px; background: var(--bg-panel); border: 1px solid var(--border);
    color: var(--text-muted); padding: 4px 10px; border-radius: 5px; cursor: pointer;
  }}
  .lang-btn:hover {{ color: var(--text); }}
  .lang-btn.active {{ color: var(--blue); border-color: var(--blue); }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 16px; margin-top: 14px; }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; font-size: 12.5px; color: var(--text-muted); }}
  .legend-swatch {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
  .meta-bar {{ display: flex; flex-wrap: wrap; gap: .5rem; margin: .9rem 0 0; }}
  .chip {{
    font-family: ui-monospace, "SFMono-Regular", Consolas, monospace;
    font-size: 12.5px; background: var(--bg-panel); border: 1px solid var(--border);
    color: var(--text-muted); padding: 4px 10px; border-radius: 5px;
  }}
  .chip strong {{ color: var(--text); font-weight: 600; }}
  .chip.status-ok strong {{ color: var(--green-text); }}
  .chip.status-pending strong {{ color: var(--amber-text); }}
  table {{
    border-collapse: collapse; width: 100%; margin-top: .5rem;
    background: var(--bg-panel); border: 1px solid var(--border); border-radius: 8px;
    overflow: hidden;
  }}
  th, td {{ border-bottom: 1px solid var(--border); padding: 8px 12px; text-align: left; font-size: 13px; vertical-align: top; }}
  th {{
    background: var(--bg-panel-alt); color: var(--text-muted); font-weight: 600;
    text-transform: uppercase; font-size: 11px; letter-spacing: .05em;
  }}
  tbody tr:last-child td {{ border-bottom: none; }}
  tbody tr:hover {{ background: var(--bg-panel-alt); }}
  td {{ font-family: ui-monospace, "SFMono-Regular", Consolas, monospace; }}
  td.rationale-cell {{
    font-family: inherit; white-space: normal; word-break: break-word;
    max-width: 380px; color: var(--text);
  }}
  tr.warmup-group-row:hover {{ background: none; }}
  td.warmup-group-cell {{ padding: 0; }}
  .warmup-details {{ padding: 10px 12px; }}
  .warmup-details > summary {{
    cursor: pointer; list-style: none; font-family: ui-monospace, "SFMono-Regular", Consolas, monospace;
    font-size: 13px; color: var(--text-muted);
  }}
  .warmup-details > summary::-webkit-details-marker {{ display: none; }}
  .warmup-details > summary::before {{ content: "\\25B6  "; color: var(--blue); }}
  .warmup-details[open] > summary::before {{ content: "\\25BC  "; }}
  .warmup-details[open] > summary {{ color: var(--text); margin-bottom: 8px; }}
  table.warmup-inner-table {{ margin: 0; }}
  .panel {{ background: var(--bg-panel); border: 1px solid var(--border); border-radius: 8px; padding: 16px 20px; }}
  .calib-warning {{ background: var(--amber-bg); border: 1px solid var(--amber-border); color: var(--amber-text); padding: 12px 16px; border-radius: 8px; }}
  .calib-ok {{ background: var(--green-bg); border: 1px solid var(--green-border); color: var(--green-text); padding: 12px 16px; border-radius: 8px; }}
  .phase-box {{ background: var(--bg-panel); border: 1px solid var(--border); border-radius: 8px; padding: 14px 18px; }}
  .phase-title {{ font-weight: 600; color: var(--text); margin-bottom: 6px; }}
  .phase-warmup .phase-title {{ color: var(--blue); }}
  .phase-bar {{ height: 6px; background: var(--bg-panel-alt); border-radius: 3px; overflow: hidden; margin: 8px 0 10px; }}
  .phase-bar-fill {{ height: 100%; background: var(--blue); }}
  .phase-note {{ color: var(--text-muted); font-size: 13px; margin: 0; }}
  .summary {{ color: var(--text-muted); font-size: 13.5px; margin: .6rem 0 0; }}
</style>
</head>
<body>
<div class="header-row">
  <h1>AutoCIM-Agent report</h1>
  <div class="lang-toggle" role="group" aria-label="Language / 언어">
    <button type="button" class="lang-btn active" data-setlang="en" onclick="setDashboardLang('en')">EN</button>
    <button type="button" class="lang-btn" data-setlang="ko" onclick="setDashboardLang('ko')">한국어</button>
  </div>
</div>
<div class="meta-bar">
  <span class="chip">model_id <strong>{html.escape(str(state.get("model_id")))}</strong></span>
  <span class="chip">hw_spec_id <strong>{html.escape(str(state.get("hw_spec_id")))}</strong></span>
  <span class="chip">iteration_count <strong>{state.get("iteration_count")}</strong></span>
  <span class="chip {"status-ok" if converged else "status-pending"}">is_converged <strong>{converged}</strong></span>
</div>
<p class="summary">{_bi(llm_summary_en, llm_summary_ko)}</p>

<h2>{_bi("Search progress", "검색 진행 상황")}</h2>
{_search_phase_summary_html(rows)}

<h2>{_bi("Calibration / precision confidence", "캘리브레이션 / 정밀도 신뢰도")}</h2>
{_calibration_section_html(state)}

<h2>{_bi("Pareto front movement (energy vs accuracy, blue = current rank 1)", "파레토 프론트 변화 (에너지 vs 정확도, 파랑 = 현재 랭크 1)")}</h2>
<div class="panel">
{_svg_pareto_chart(rows)}
{_pareto_legend_html()}
</div>

<h2>{_bi("Per-iteration candidates", "반복별 후보 목록")}</h2>
<table>
<thead><tr>
<th>{_bi("iter", "반복")}</th>
<th>{_bi("search phase", "탐색 단계")}</th>
<th>{_bi("accuracy", "정확도")}</th>
<th>energy_pj</th>
<th>latency_ms</th>
<th>pareto_rank</th>
<th>{_bi("converged", "수렴 여부")}</th>
<th>{_bi("rationale", "근거")}</th>
</tr></thead>
<tbody>
{_table_rows_html(rows)}
</tbody>
</table>
<script>
function setDashboardLang(lang) {{
  document.querySelectorAll('[data-lang]').forEach(function (el) {{
    el.hidden = el.getAttribute('data-lang') !== lang;
  }});
  document.querySelectorAll('.lang-btn').forEach(function (btn) {{
    btn.classList.toggle('active', btn.dataset.setlang === lang);
  }});
}}
</script>
</body>
</html>"""
