"""Task: generate eval spec and score it against the original function.

This is the core logic that GEPA calls for each candidate prompt:
1. Create an agent with the candidate's eval_spec_context as system prompt
2. Generate eval YAML for a function
3. Post-validate expected values against the real function
4. Self-correct failing cases via a retry
5. Return detailed scoring and failure diagnostics
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal

import logfire
import yaml
from pydantic_ai import Agent
from vowel.eval_types import EvalsSource
from vowel.runner import RunEvals
from vowel.tdd import TDDGenerator
from vowel.validation import validate_and_fix_spec

from .functions import FunctionCase

MODEL = "openrouter:google/gemini-3-flash-preview"


@dataclass
class CaseFailure:
    """A single failing case with diagnosis."""

    case_id: str
    evaluator: str
    reason: str
    category: (
        str  # WRONG_EXPECTED, INVENTED_RAISES, FORMAT_MISMATCH, OVER_STRICT_ASSERTION, BAD_INPUT
    )


@dataclass
class EvalResult:
    """Result of generating + running evals for one function."""

    func_name: str
    yaml_spec: str | None = None
    total_cases: int = 0
    passed_cases: int = 0
    pass_rate: float = 0.0
    failures: list[CaseFailure] = field(default_factory=list)
    error: str | None = None

    @property
    def score(self) -> float:
        """Primary optimization score: pass rate."""
        if self.error:
            return 0.0
        return self.pass_rate

    def feedback_text(self) -> str:
        """Human-readable feedback for the proposer LLM."""
        lines = [f"Function: {self.func_name}"]
        lines.append(
            f"Score: {self.pass_rate:.0%} ({self.passed_cases}/{self.total_cases} assertions)"
        )

        if self.error:
            lines.append(f"ERROR: {self.error}")
            return "\n".join(lines)

        if not self.failures:
            lines.append("All cases passed!")
            return "\n".join(lines)

        lines.append("Failures:")
        for f in self.failures:
            lines.append(f"  [{f.category}] {f.case_id}: {f.evaluator} — {f.reason}")

        return "\n".join(lines)


def _diagnose_failure(
    case_data: dict,
    assertion_name: str,
    assertion_info: dict,
    func_case: FunctionCase,
) -> CaseFailure:
    """Classify a single assertion failure into a category."""
    reason = assertion_info.get("reason", "")
    case_id = case_data.get("case", "unknown")

    # Raises failures
    if "Raises" in assertion_name:
        if "returned normally" in reason:
            return CaseFailure(
                case_id=case_id,
                evaluator=assertion_name,
                reason=reason,
                category="INVENTED_RAISES",
            )
        if "but got" in reason:
            return CaseFailure(
                case_id=case_id,
                evaluator=assertion_name,
                reason=reason,
                category="WRONG_EXCEPTION_TYPE",
            )

    # EqualsExpected failures — run original function to compare
    if "EqualsExpected" in assertion_name:
        return CaseFailure(
            case_id=case_id,
            evaluator=assertion_name,
            reason=reason,
            category="WRONG_EXPECTED",
        )

    # Type failures
    if "Type" in assertion_name and "Invalid type expression" in reason:
        return CaseFailure(
            case_id=case_id,
            evaluator=assertion_name,
            reason=reason,
            category="UNSUPPORTED_TYPE_EXPR",
        )

    # Assertion failures
    if "Assertion" in assertion_name:
        return CaseFailure(
            case_id=case_id,
            evaluator=assertion_name,
            reason=reason,
            category="OVER_STRICT_ASSERTION",
        )

    return CaseFailure(
        case_id=case_id,
        evaluator=assertion_name,
        reason=reason,
        category="OTHER",
    )


# ──────────────────────────────────────────────────────────────────────
# Post-validation: heuristic checks on generated YAML before scoring
# ──────────────────────────────────────────────────────────────────────


def _post_validate_expected(yaml_spec: str, func_case: FunctionCase) -> str:
    """Run real function on each case's inputs and fix wrong expected values.

    Only fixes `expected` fields where the function returns normally.
    Leaves assertions, raises, and cases without expected alone.
    Uses targeted regex replacement to preserve YAML formatting (inline lists etc.).
    Returns the (potentially modified) yaml_spec.
    """
    try:
        data = yaml.safe_load(yaml_spec)
    except Exception:
        return yaml_spec

    if not isinstance(data, dict):
        return yaml_spec

    # Find the eval spec for this function
    spec = data.get(func_case.name)
    if not spec or not isinstance(spec, dict):
        return yaml_spec

    dataset = spec.get("dataset", [])
    if not dataset:
        return yaml_spec

    # Collect fixes: list of (case_id, old_expected, new_actual)
    fixes: list[tuple[str, object, object]] = []

    for item in dataset:
        case = item.get("case", {}) if isinstance(item, dict) else {}
        if not isinstance(case, dict):
            continue

        # Skip cases that test raises or use assertion instead of expected
        if "raises" in case or "expected" not in case:
            continue

        case_id = case.get("id", "")

        # Get inputs
        inputs = case.get("inputs") or case.get("input")
        if inputs is None:
            continue

        # Run the real function
        try:
            if "inputs" in case and isinstance(inputs, list):
                actual = func_case.func(*inputs)
            else:
                actual = func_case.func(inputs)
        except Exception:
            continue

        # Compare: fix if expected is wrong
        expected = case["expected"]
        try:
            if actual != expected:
                if str(actual) == str(expected):
                    continue
                fixes.append((case_id, expected, actual))
        except Exception:
            continue

    if not fixes:
        return yaml_spec

    # Apply fixes via targeted regex replacement in the YAML string
    result = yaml_spec
    for case_id, _old_expected, new_actual in fixes:
        safe_val = _yaml_safe_value(new_actual)
        # Format the replacement value for YAML
        if isinstance(safe_val, str):
            # Quote strings to be safe
            yaml_val = repr(safe_val)
        elif isinstance(safe_val, list):
            # Use compact flow-style for lists
            yaml_val = yaml.dump(safe_val, default_flow_style=True).strip()
        elif isinstance(safe_val, dict):
            yaml_val = yaml.dump(safe_val, default_flow_style=True).strip()
        else:
            yaml_val = str(safe_val)

        # Find the expected line within this case's block (after the case id line)
        # Pattern: after "id: <case_id>", find the next "expected:" line and replace value
        pattern = re.compile(
            rf"(id:\s*{re.escape(case_id)}.*?expected:\s*)(.+)",
            re.DOTALL,
        )
        match = pattern.search(result)
        if match:
            result = result[: match.start(2)] + yaml_val + result[match.end(2) :]

    return result


def _yaml_safe_value(val):
    """Convert a Python value to a YAML-friendly representation."""
    if isinstance(val, Decimal):
        return str(val)
    if isinstance(val, (list, tuple)):
        return [_yaml_safe_value(v) for v in val]
    if isinstance(val, dict):
        return {k: _yaml_safe_value(v) for k, v in val.items()}
    return val


# ──────────────────────────────────────────────────────────────────────
# Self-correction: retry failed cases
# ──────────────────────────────────────────────────────────────────────


def _self_correct(
    yaml_spec: str,
    failures: list[dict],
    agent: Agent,
    signature_context: str,
) -> str | None:
    """Ask the LLM to fix specific failing cases in the eval YAML.

    Returns corrected yaml_spec, or None if correction failed.
    """
    if not failures:
        return None

    # Format up to 5 failures for the correction prompt
    failure_lines = []
    for f in failures[:5]:
        failure_lines.append(
            f"- Case '{f['case_id']}': evaluator '{f['evaluator']}' failed — {f['reason']}"
        )
    failures_text = "\n".join(failure_lines)

    correction_prompt = f"""The following eval YAML was generated but some cases FAILED when run against the real function.
Fix ONLY the failing cases. Do not change passing cases. Return the complete corrected YAML.

## Function Signature
{signature_context}

## Current YAML (with failures)
```yaml
{yaml_spec}
```

## Failures
{failures_text}

## Instructions
- If an `expected` value is wrong, either correct it or replace with an `assertion`
- If a `raises` case failed because the function returned normally, remove the `raises` field and add expected/assertion
- If a `raises` case has the wrong exception type, fix the type or use `?` suffix for soft match
- Keep all passing cases unchanged
- Return ONLY the corrected YAML, no explanation
"""
    try:
        result = agent.run_sync(correction_prompt, output_type=EvalsSource)
        corrected = result.output.yaml_spec
        # Basic sanity: must parse as YAML
        yaml.safe_load(corrected)
        return corrected
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────
# Scoring helper
# ──────────────────────────────────────────────────────────────────────


def _score_summary(summary, func_case: FunctionCase) -> tuple[int, int, list[CaseFailure]]:
    """Extract per-assertion scores from a run summary."""
    total_assertions = 0
    passed_assertions = 0
    failures = []

    for r in summary.results:
        if r.report:
            for case in r.report.cases:
                for name, info in case.assertions.items():
                    total_assertions += 1
                    if info.value:
                        passed_assertions += 1
                    else:
                        failure = _diagnose_failure(
                            {"case": case.name},
                            name,
                            {"reason": str(info.reason) if info.reason else ""},
                            func_case,
                        )
                        failures.append(failure)

    return total_assertions, passed_assertions, failures


def generate_and_score(
    func_case: FunctionCase,
    eval_spec_context: str,
    model: str = MODEL,
) -> EvalResult:
    """Generate eval spec for a function and score it against the original.

    Pipeline:
    1. Generate signature via TDD
    2. Generate eval YAML with few-shot examples
    3. Post-validate expected values by running real function
    4. Score against original
    5. If failures exist, self-correct and re-score

    Args:
        func_case: The function to generate evals for
        eval_spec_context: The candidate EVAL_SPEC_CONTEXT prompt to use
        model: LLM model for eval generation

    Returns:
        EvalResult with score and failure diagnostics
    """
    result = EvalResult(func_name=func_case.name)

    with logfire.span("generate_and_score", func_name=func_case.name):
        try:
            # 1. Generate signature via TDD
            with logfire.span("generate_signature", func_name=func_case.name):
                gen = TDDGenerator(model=model)
                signature = gen.generate_signature(func_case.description, func_case.name)
                logfire.info(
                    "signature_generated",
                    func_name=func_case.name,
                    params=str(signature.params),
                )

            # 2. Build eval agent with CANDIDATE prompt
            system_prompt = f"""You are an expert test case generator.
Your task is to generate comprehensive eval specs from function signatures.

{eval_spec_context}

═══════════════════════════════════════════════════════════════════════════
CRITICAL RULES - READ CAREFULLY
═══════════════════════════════════════════════════════════════════════════

## 1. INPUT FORMAT - ALWAYS USE INLINE LIST FORMAT

ALWAYS use `inputs:` with an INLINE LIST `[arg1, arg2]`, NEVER YAML list syntax with dashes.

✅ CORRECT (inline list on same line):
```yaml
inputs: [{{"a": 1, "b": 2}}, "a.b"]
inputs: ["hello world", true]
inputs: [[1, 2, 3], 5]
```

❌ WRONG (YAML list with dashes - breaks parsing):
```yaml
inputs:
  - {{"a": 1}}
  - "path"
```

For single argument, use `input:` (singular):
```yaml
input: "2 + 3 * 4"
input: [1, 2, 3, 4, 5]
```

## 2. ASSERTION VARIABLES

In assertions, access inputs positionally:
- `input` - the raw input (for single `input:` field)
- `input[0]`, `input[1]` - positional args (for `inputs:` list)
- `output` - function return value
- `expected` - expected value if specified

## 3. EXPECTED VALUES - CALCULATE CAREFULLY!

⚠️ DO NOT GUESS expected values! Trace through the algorithm mentally.
If unsure, use `assertion` instead of `expected`.
"""

            eval_agent = Agent(
                model,
                output_type=EvalsSource,
                system_prompt=system_prompt,
            )
            # Override the generator's eval agent with our custom one
            gen._eval_agent = eval_agent

            # 3. Generate evals (with retry on YAML parse failure)
            with logfire.span("generate_eval_yaml", func_name=func_case.name):
                sig_context = signature.to_prompt_context()
                prompt = f"""Generate eval YAML spec for this function signature:

{sig_context}

Requirements:
- Use `{signature.name}` as eval_id
- Generate at least 8 diverse test cases
- Include normal cases, edge cases, and error cases
- Test all parameters and return type
- Add appropriate global evaluators (type checks, assertions)

IMPORTANT: In assertions, use `input[0]`, `input[1]` to access positional args.
"""
                yaml_spec = None
                last_error = None
                for attempt in range(2):
                    try:
                        eval_result = gen.eval_agent.run_sync(prompt, output_type=EvalsSource)
                        candidate = eval_result.output.yaml_spec

                        # Sanitize YAML tags
                        candidate = re.sub(r"!!python/[\w.:]+", "", candidate)
                        candidate = re.sub(r"!!binary\b", "", candidate)

                        # Validate + fix
                        yaml.safe_load(candidate)
                        validation = validate_and_fix_spec(candidate)
                        if validation.was_modified:
                            candidate = validation.fixed_yaml

                        yaml_spec = candidate
                        break
                    except Exception as e:
                        last_error = e
                        logfire.info(
                            "yaml_gen_retry",
                            func_name=func_case.name,
                            attempt=attempt,
                            error=str(e),
                        )

                if yaml_spec is None:
                    raise last_error  # type: ignore[misc]

                logfire.info(
                    "eval_yaml_generated",
                    func_name=func_case.name,
                    yaml_len=len(yaml_spec),
                )

            # 4. Post-validate: fix wrong expected values by running the real function
            with logfire.span("post_validate", func_name=func_case.name):
                yaml_spec = _post_validate_expected(yaml_spec, func_case)

            result.yaml_spec = yaml_spec

            # 5. Run evals against ORIGINAL function
            with logfire.span("run_evals_against_original", func_name=func_case.name):
                runner = RunEvals.from_source(yaml_spec)
                runner = runner.with_functions({func_case.name: func_case.func})
                runner = runner.ignore_duration()
                summary = runner.run()

                total_assertions, passed_assertions, failures = _score_summary(summary, func_case)

            # 6. Self-correction: if there are failures, ask LLM to fix them
            if failures and total_assertions > 0:
                fail_ratio = len(failures) / total_assertions
                # Only self-correct if <30% failures (not totally broken)
                if fail_ratio < 0.30:
                    with logfire.span(
                        "self_correct",
                        func_name=func_case.name,
                        num_failures=len(failures),
                    ):
                        failure_dicts = [
                            {
                                "case_id": f.case_id,
                                "evaluator": f.evaluator,
                                "reason": f.reason,
                            }
                            for f in failures
                        ]
                        corrected = _self_correct(yaml_spec, failure_dicts, eval_agent, sig_context)
                        if corrected:
                            # Re-validate
                            try:
                                corrected = re.sub(r"!!python/[\w.:]+", "", corrected)
                                corrected = re.sub(r"!!binary\b", "", corrected)
                                yaml.safe_load(corrected)
                                v2 = validate_and_fix_spec(corrected)
                                if v2.was_modified:
                                    corrected = v2.fixed_yaml

                                # Post-validate again
                                corrected = _post_validate_expected(corrected, func_case)

                                # Re-score
                                runner2 = RunEvals.from_source(corrected)
                                runner2 = runner2.with_functions({func_case.name: func_case.func})
                                runner2 = runner2.ignore_duration()
                                summary2 = runner2.run()

                                t2, p2, f2 = _score_summary(summary2, func_case)

                                # Keep corrected version only if it's better
                                if t2 > 0 and (p2 / t2) > (passed_assertions / total_assertions):
                                    logfire.info(
                                        "self_correct_improved",
                                        func_name=func_case.name,
                                        old_pass=f"{passed_assertions}/{total_assertions}",
                                        new_pass=f"{p2}/{t2}",
                                    )
                                    total_assertions, passed_assertions, failures = (
                                        t2,
                                        p2,
                                        f2,
                                    )
                                    result.yaml_spec = corrected
                                else:
                                    logfire.info(
                                        "self_correct_no_improvement",
                                        func_name=func_case.name,
                                    )
                            except Exception as e:
                                logfire.info(
                                    "self_correct_failed",
                                    func_name=func_case.name,
                                    error=str(e),
                                )

            # 7. Finalize scores
            result.failures = failures
            result.passed_cases = passed_assertions
            result.total_cases = total_assertions
            result.pass_rate = passed_assertions / total_assertions if total_assertions > 0 else 0.0

            logfire.info(
                "scoring_complete",
                func_name=func_case.name,
                pass_rate=result.pass_rate,
                passed=passed_assertions,
                total=total_assertions,
                num_failures=len(result.failures),
                failure_categories={f.category for f in result.failures},
            )

        except Exception as e:
            result.error = f"{type(e).__name__}: {e}"
            logfire.warn("generate_and_score failed", func=func_case.name, error=str(e))

    return result
