<!-- AGENTS.md -->
# Agent instructions for Parley

## Trigger phrases
- "resume" → run the Resume protocol, then stop after the brief and plan.
- "resume go" → run the Resume protocol, then execute the next action
  (only if it is an unchecked item under "Current phase"; otherwise ask).
- "wrap" → run the Wrap protocol.
These phrases mean the same thing in any tool (Claude Code, Codex, Antigravity).
## Always
- Read `PROJECT_CONTEXT.md` before any work. Its "Hard rules" section is binding.
- Only one agent edits code and context files per session. If you are a
  reviewer agent, do not write files.

## Resume protocol (run when asked to "resume")
1. Read, in order: `PROJECT_CONTEXT.md`, `docs/session-handoff.md`,
   the last 30 lines of `docs/work-log.md`, `git status`, `git log --oneline -10`.
2. Reconcile. Report any contradiction between these sources, for example:
   handoff says a step is in progress but git shows it committed; the stack
   section disagrees with config; uncommitted changes the handoff doesn't explain.
3. Print a brief (max 15 lines):
   - Phase and gate status
   - Last completed step
   - In-progress step (exact file, function, command)
   - Next action (exactly one)
   - Blockers
   - Pending confirmations (findings waiting on user OK before writing to logs)
   - Contradictions found
4. Propose the next action as small, testable steps (command + how to verify).
5. Execute only if the user said "resume go" AND the next action is an
   unchecked item under "Current phase" in PROJECT_CONTEXT.md. Anything else:
   stop and ask.

## While working (checkpoint rule)
- After each completed step, overwrite the "In progress" and "Next action"
  sections of `docs/session-handoff.md`. Sessions end abruptly; the handoff
  must be correct at any moment.
- Commit after each verified step with a message that names the step.
- New measurements go under "Pending confirmations" in the handoff, never
  directly into decisions-log / work-log / latency-budget (hard rule 4).

## Wrap protocol (run when asked to "wrap")
1. Rewrite `docs/session-handoff.md` fully (template in that file).
2. Propose a diff to `PROJECT_CONTEXT.md` (checkboxes, open questions,
   stack) and wait for approval before applying.
3. List pending log entries in the exact one-line format, and wait for approval.
4. Confirm `git status` is clean or explain what is uncommitted and why.