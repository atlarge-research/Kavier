---
name: writing-tests
description: Use when writing, adding, or reviewing tests in this repo (pytest) — before writing any assert. Triggers on "add a test", "get coverage up", "be thorough/exhaustive", a new test file, a growing parametrize grid, or any assertion that pins a magic number or only checks not-None / > 0 / isinstance / "runs without error".
---

# Writing Tests

## The one rule: every assertion needs an independent oracle

**A test's expected value must come from a source independent of the code under test.**
If the expected value is produced by the code you're testing (call it, then assert it equals
what it returned) or by the *same formula* the code uses, the test passes *by construction* —
it re-encodes current behavior instead of verifying correct behavior. That is not a test. It is
a snapshot with an assertion costume.

The shape (`assert x == pytest.approx(400.0)`) looks identical whether the `400.0` is gold or
garbage. What separates them is **where the number came from**. Compare — both from this repo:

```python
# ✅ GOOD — oracle hand-derived from first principles, shown in the comment
# tests/test_inference/test_energy_unit_regressions.py
# 100 Wh over 250_000 tokens. By hand: 100 / 250_000 * 1e6 = 400 Wh/Mtoken.
assert eff == pytest.approx(400.0)

# ❌ TAUTOLOGICAL — 0.4 / 0.77 are just what the function printed last run; no derivation
# tests/test_inference/test_timings.py
assert math.isclose(t1, 0.4, rel_tol=0.05)
assert math.isclose(t2, 0.77, rel_tol=0.05)
```

## Where a legitimate oracle comes from

Pick at least one. The comment next to the assert must say which, and show the arithmetic.

| Oracle source | Example in this repo |
|---|---|
| **Hand-derived from first principles** | `# 100 / 250_000 * 1e6 = 400` (`test_energy_unit_regressions.py`) |
| **A known/analytic reference point** | u=0 → idle power, u=1 → max power (`test_stage_invariants.py` endpoints) |
| **A simpler special case done by hand** | r=1 collapses `2u − u**r` to `u`, so oracle is `idle + (max−idle)*u` — *different form* from the impl |
| **An invariant / property** (not a value) | monotonic, bounded `[idle,max]`, `min_rule ≤ left_step`, KV util ∈ [0,1] (`test_stage_invariants.py`, `test_emissions.py`) |
| **A scaling law** | 2× output with no KV ⇒ ~4× time, `n(n+1)/2 ~ n²` (`test_decode_no_kv_quadratic_ratio`) |
| **Cross-check vs. a different method or a pinned bug value** | `assert wh != pytest.approx(7.2)  # the old /1000 bug` |

The tell for a *good* derived oracle: it is written in a **different form** than the
implementation. `test_nonlinear_exponent_matches_closed_form` recomputes `2u - u**2` verbatim
from the source → tautological. The r=1 test simplifies the algebra by hand first → real.

## Not a test (even when green)

- **Smoke asserts as the only check**: `is not None`, `> 0`, `isinstance(x, float)`,
  `assert result` — passes if the function returns any positive constant. Fine as *one* line
  guarding a real assertion; never the whole test. (`test_return_type_is_float`,
  `test_efficiency_is_finite_when_sized`.)
- **Named for a behavior it can't catch**: `test_multi_gpu_increases_throughput` asserting
  `r4 > r1 * 0.5` — passes even if 4 GPUs were *slower*. The assertion must be able to fail on
  the exact regression the name promises.
- **Magic catalog constants**: `assert len(model_names()) == 16`. Assert the list *matches the
  spec library* (an invariant), not a number that breaks every time someone adds a GPU.
- **Over-mocking**: mocking the thing under test, or so much that you only test the mock. (This
  repo barely mocks — keep it that way; only fake the *environment*, e.g. `sys.argv`, `isatty`.)

## Every test must earn its place

Count the code's **distinguishable behaviors** first; that number is your test budget. For the
5-line `mse_power` there are ~8: two endpoints, the linear ramp, the max(compute, mem) rule, two
clamps, the r≠1 branch, the bounds invariant. An unguided agent once wrote 57 tests for it — the
extra 49 added runtime and maintenance surface, zero detection power.

- **One behavior per test**, and the name states it as a specification
  (`test_negative_utilization_clamped_to_idle`). If two tests can only fail together, delete one.
- **A parametrize grid must vary the behavior, not just the number.** Once endpoints plus one
  interior point pin a linear ramp, adding u=0.25/0.5/0.75 detects nothing new. Parametrize over
  things that *differ* (the whole GPU×LLM catalog — each spec is a distinct case); for "holds on
  a whole range", write one Hypothesis property instead of a hand-picked grid.
- **Never write a test whose purpose is to exist** — raising a count, a coverage %, or the
  appearance of rigor. Coverage is a byproduct of testing behaviors; a covered line with no
  falsifiable assert is not tested. "Exhaustive" means every *behavior* rejected, not every
  input enumerated.

## The falsification check (do this before committing any test)

For each test, ask: **"What bug would make this fail?"** Then mentally break the code:
- Replace the function body with `return <a constant>` — does the test go red? If not, it's smoke.
- Perturb the formula (drop a term, change `/3600` to `/1000`) — does it go red? If not, the
  oracle isn't independent.

If you can't name a specific wrong behavior the test rejects, the test verifies nothing. Delete
or fix it — a green test that can't fail is worse than no test, because it reads as coverage.

## Choosing the test type (this repo's four layers)

- **Invariant** — physical/logical laws (monotonicity, non-negativity, `power ∈ [idle,max]`,
  fragments tile duration). Parametrize over the *whole* GPU×LLM catalog.
- **Regression** — cite the commit/issue in the docstring and hand-derive the expected value;
  where useful, also assert `!= <old buggy value>`.
- **Property-based** — Hypothesis for "holds for all inputs in a range" (see `test_stage_invariants.py`).
- **Integration** — shell out to the real CLI via `python -m kavier.cli <subcommand>`.

Prefer an **invariant or a hand-derived value** over pinning a raw engine output. A `rel=1e-6`
snapshot of `tokens_per_second == 3090.592817` fails on *any* change, correct or not, and tells
you nothing about *why*. If you must pin an engine output, derive it or explain in the comment
what physical quantity it represents and why that exact value is right.

## Craft standards (state of the art, condensed)

- **Arrange–Act–Assert**, and no logic in tests: an `if`, or a loop/formula that *computes the
  expected value*, means the oracle isn't independent — derive it by hand or assert a property.
- **Test the contract, not the implementation**: assert what the function promises (endpoints,
  clamping, monotonicity, units), never its internals or an incidental output pinned at
  `rel=1e-6`. Contract tests survive correct refactors and fail on real regressions; snapshots
  do exactly the opposite.
- **Deterministic**: no wall clock, no unseeded randomness, no dependence on test order or on
  another test's state. Randomized input exploration belongs in Hypothesis, which shrinks
  failures and replays them.
- **The falsification check is mutation testing done in your head.** For high-value modules,
  run it for real (`mutmut run --paths-to-mutate src/...`) instead of trusting green.

## Red flags — you're about to write a fake test

- You ran the code, copied the number it printed, and pasted it into `assert == <number>`.
- Your `expected = ...` line is the same formula as the source (or calls the function under test).
- The only assertions are `not None`, `> 0`, `isinstance`, or "it didn't raise".
- You can't answer "what bug would make this fail?"
- The test name promises a *direction* or *magnitude* the assertion doesn't actually pin.
- You're adding a test and can't say which *behavior* it covers that the existing ones don't.

**All of these mean: stop, find an independent oracle, and write the derivation in a comment.**

## Rationalizations

| Excuse | Reality |
|---|---|
| "It proves the function works." | It proves the function does what it currently does. A tautology can't distinguish right from wrong. |
| "I need coverage green fast." | Coverage of untested behavior is a false signal — worse than a known gap. Derive one real assert; it's ~2 minutes. |
| "The number came from a real run, so it's real." | Real *output* ≠ *correct* output. The oracle must be independent of the run. |
| "It's just a smoke/type check." | One guard line is fine; a whole test of only smoke asserts verifies nothing. |
| "Deriving the value by hand is overthinking it." | The derivation *is* the test. Without it you've written an assertion, not a verification. |
| "More tests = more safety." | Detection power comes from behaviors rejected, not test count. 57 tests of one linear ramp catch the same bugs as 3 — and cost 19x the maintenance. |
| "The user asked for exhaustive coverage." | Exhaustive over *behaviors*, not inputs. Enumerate the branches/clamps/terms; one falsifiable test each. |
