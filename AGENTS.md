- Do not preserve backward compatibility. Remove obsolete paths instead of
  adding compatibility layers, fallbacks, or migrations.
- Choose the simplest implementation that fully meets the current
  requirements. Avoid speculative abstractions, configuration, and
  indirection.
- Grow the system in layers. Start from the smallest version that works end
  to end, and add each new capability on top of a product that already
  works. Never trade a working product for unfinished complexity.
- Keep components modular and concerns clearly separated.
- Prefer established, well-maintained libraries when they reduce overall
  complexity or improve reliability. Do not reimplement common
  functionality without a clear reason.
- Lean on the dependencies already in the project before writing your own
  implementation or adding packages. Do not assume a library lacks a
  capability without checking its documentation and types.
- Make architectural decisions for the long term. Do not accept a stopgap
  that only works for now and is meant to be replaced later.

## Testing principles

- Test business behavior and public contracts, not implementation details. Assert stable outcomes, invariants, and meaningful boundaries; do not assert incidental exact values unless the requirement explicitly says so. For example, inventory should be `>= 0`, not arbitrarily equal to `5`. Avoid assertions about private methods, call counts, or internal sequencing unless they are observable business behavior. If a harmless refactor breaks a test, reconsider the test first.
- Use BDD and write the acceptance scenario before implementation. Every scenario description starts with `User` and uses `Given / When / Then` so that a user or product owner can understand it. Do not implement first and then write tests that merely reproduce the implementation.
- Do not use mocks. Prefer real Redis, PostgreSQL, and other in-scope dependencies. When an external system cannot be used directly, inject a fake and verify that fake against the real system's contract. Use fake clocks/timers for time-dependent behavior; never sleep for a few seconds and hope the test is reliable.
- Prefer end-to-end tests that start at the user entry point and verify the visible result. Use lower-level tests only where an E2E test is unsuitable, and keep the boundary explicit. During iteration run the smallest focused scope affected by the change; run the full suite at integration, pre-merge, and release checkpoints. Report unrun external boundaries such as ComfyUI, GPU, LLM, or social providers instead of implying that local tests proved them.
- Treat branch coverage as a diagnostic target, with `90%` as the default goal after behavior-oriented tests are sound. Investigate uncovered branches, especially failure paths, and document why important branches remain untested. Never add brittle or tautological tests merely to raise the percentage.
- Enforce mechanical parts of these rules with lint or static checks where practical. Keep tests focused on business behavior and contracts that tooling cannot infer.
