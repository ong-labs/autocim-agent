# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 5. AutoCIM-Agent Specific Rules

### A. State Management & Reducers
- **Strict Schema Adherence**: Always respect `AutoCIMState` definitions. Do not add arbitrary keys to the state unless specified in Pydantic/TypedDict schemas.
- **Namespaced Metrics**: When updating `metrics_store`, always nest outputs under the node's namespace (e.g., `metrics_store: {"mapper": {"status": "SUCCESS", "data": ...}}`) to prevent cross-agent race conditions during parallel execution.
- **State Sanitization**: Ensure `@planner` cleans up `human_overrides` and resets transient flags (`needs_hitl`, `retry_count`) immediately after consumption.

### B. LangGraph Execution & HITL
- **v0.2+ Interrupt Standard**: Do not use `interrupt_before` in `compile()`. Use dynamic `interrupt()` calls inside nodes or pass `Command(resume=...)` to handle Human-in-the-Loop flow cleanly.
- **No Double Resumes**: Never mix `graph.update_state()` with dynamic `interrupt()` in a way that duplicates state updates.

### C. Config-Driven Hardware & Model Abstraction
- **No Hardcoded Specs**: Never hardcode hardware parameters (e.g., `128x128 Crossbar`, `ADC/DAC bits`) or target neural network architectures directly into node logic. Load them dynamically via `HWConfig` or `ExecutionContext` metadata.
- **Model Agnostic**: Keep `@tuner` and `@mapper` layer-traversal logic general enough to handle arbitrary PyTorch `nn.Module` or ONNX graphs.

### D. Tool Interceptor & External Solvers
- **Tool Interceptor Protocol**: All backend simulator/PyTorch tools must be wrapped or intercepted using `wrap_tool_call` to inject `ExecutionContext` and handle exceptions without breaking the agent loop.
- **Mock First**: For external simulators (NeuroSim, CIM-Loop), implement clear mock data interfaces first before attempting binary or C-Extension integrations.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
