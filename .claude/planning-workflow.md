# Planning Workflow — Multi-Agent Markdown Planning

When the user gives a planning topic ("plan a spec for X", "draft a design for Y"), you are the **orchestrator**. Route work through three subagents defined in `.claude/agents/` and stop when the critic approves or after 3 revision rounds. Chain calls without asking the user between steps, except at the Phase 1 skeleton gate (see Phase 1 rules).

## Agents

- **planner** (`.claude/agents/planner.md`, Opus) — produces the skeleton: sections, scope, open questions. Output → `.claude/plan/01-skeleton.md`.
- **researcher** (`.claude/agents/researcher.md`, Sonnet) — drafts content from the skeleton; also applies critic revisions. Output → `.claude/plan/02-draft.md` (overwritten each round).
- **critic** (`.claude/agents/critic.md`, Opus) — reviews the current draft. MUST end its output with a final line `STATUS: APPROVED` or `STATUS: REVISE`, followed (if REVISE) by a YAML block of numbered revisions.

## Flow

1. **Skeleton.** Invoke `planner` with the user's topic. Write to `.claude/plan/01-skeleton.md`. Then **stop the orchestrator turn** and report the path to the user. Resume on the user's next message — "go", "proceed", or explicit edits to the skeleton. If the user's original request contained "walk away", "auto-approve", or "don't stop", skip this gate and proceed directly to Phase 2. (Claude Code has no time-based resume; "pause" means return control.)
2. **Draft.** Invoke `researcher` with `.claude/plan/01-skeleton.md` as an explicit Read input. Write to `.claude/plan/02-draft.md`.
3. **Review loop** (max 3 rounds):
   - Invoke `critic` with (a) `.claude/plan/02-draft.md` and `.claude/plan/01-skeleton.md` as Read inputs, (b) the current round number (1/2/3), and (c) from round 2 onward, the previous revision YAML inline.
   - Parse the final `STATUS:` line.
   - `APPROVED` → go to Phase 4.
   - `REVISE` → invoke `researcher` (mode B) with the revision YAML inline; overwrite `.claude/plan/02-draft.md`; next round.
   - **Round-3 critic prompt MUST prepend:** `ROUND=3 — blockers only. Do not gate approval on majors or nits. If no blockers remain, emit STATUS: APPROVED.`
4. **Finalize.** Copy the last draft to `.claude/plan/final-plan.md` only if the loop exited via `APPROVED`. If it exited via round-cap-with-blockers, stop at `02-draft.md` and surface the blockers. Report to the user: rounds used, word count, path.

## Contract rules

- **Strict critic output format.** If the critic output has no parseable `STATUS:` line, treat as REVISE and log a warning in your phase report. Do not guess approval from prose.
- **Explicit file inputs.** Subagents start fresh with no inherited context. Every subagent prompt must name the exact files to Read (e.g., `.claude/plan/01-skeleton.md`, `.claude/plan/02-draft.md`).
- **Round cap stop.** If round 3 finishes with unresolved blockers, stop at `.claude/plan/02-draft.md`, do NOT create `final-plan.md`, and surface the outstanding blockers.
- **Diff between rounds.** From round 2 onward, pass the critic the revision list it emitted last round so it reviews the delta, not the whole doc from scratch.
- **Namespace.** All outputs under `.claude/plan/`. Create the directory if absent.

## When the user should intervene

- Phase 1 skeleton assumptions look wrong → interrupt and re-prompt `planner`.
- Round 3 cap hit with real blockers remaining → run one more manual round.

Otherwise, let it run.
