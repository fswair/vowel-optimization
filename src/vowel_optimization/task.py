"""Task: generate eval spec and score it against the original function.

This is the core logic that GePa calls for each candidate prompt:
1. Create an agent with the candidate's eval_spec_context as system prompt
2. Generate eval YAML for a function
3. Run generated evals against the original implementation
4. Return detailed scoring and failure diagnostics
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import yaml
import logfire
from pydantic_ai import Agent

from vowel.ai import EvalsSource
from vowel.runner import RunEvals
from vowel.tdd import TDDGenerator
from vowel.validate_spec import validate_and_fix_spec

from .functions import FunctionCase

MODEL = "openrouter:google/gemini-3-flash-preview"


@dataclass
class CaseFailure:
    """A single failing case with diagnosis."""

    case_id: str
    evaluator: str
    reason: str
    category: str  # WRONG_EXPECTED, INVENTED_RAISES, FORMAT_MISMATCH, OVER_STRICT_ASSERTION, BAD_INPUT


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
        lines.append(f"Score: {self.pass_rate:.0%} ({self.passed_cases}/{self.total_cases} assertions)")

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


def generate_and_score(
    func_case: FunctionCase,
    eval_spec_context: str,
    model: str = MODEL,
) -> EvalResult:
    """Generate eval spec for a function and score it against the original.

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
                logfire.info("signature_generated", func_name=func_case.name, params=str(signature.params))

            # 2. Build eval agent with CANDIDATE prompt (not the default)
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

            # Override the generator's eval agent with our custom one
            gen._eval_agent = Agent(
                model,
                output_type=EvalsSource,
                system_prompt=system_prompt,
            )

            # 3. Generate evals
            with logfire.span("generate_eval_yaml", func_name=func_case.name):
                prompt = f"""Generate eval YAML spec for this function signature:

{signature.to_prompt_context()}

Requirements:
- Use `{signature.name}` as eval_id
- Generate at least 8 diverse test cases
- Include normal cases, edge cases, and error cases
- Test all parameters and return type
- Add appropriate global evaluators (type checks, assertions)

IMPORTANT: In assertions, use `input[0]`, `input[1]` to access positional args.
"""
                eval_result = gen.eval_agent.run_sync(prompt)
                yaml_spec = eval_result.output.yaml_spec

                # Sanitize YAML tags
                yaml_spec = re.sub(r'!!python/[\w.:]+', '', yaml_spec)
                yaml_spec = re.sub(r'!!binary\b', '', yaml_spec)

                # Validate + fix
                yaml.safe_load(yaml_spec)
                validation = validate_and_fix_spec(yaml_spec)
                if validation.was_modified:
                    yaml_spec = validation.fixed_yaml

                result.yaml_spec = yaml_spec
                logfire.info("eval_yaml_generated", func_name=func_case.name, yaml_len=len(yaml_spec))

            # 4. Run evals against ORIGINAL function
            with logfire.span("run_evals_against_original", func_name=func_case.name):
                runner = RunEvals.from_source(yaml_spec)
                runner = runner.with_functions({func_case.name: func_case.func})
                runner = runner.ignore_duration()
                summary = runner.run()

                # 5. Collect results — score per-assertion (not per-case)
                #    This avoids a single bad global evaluator zeroing out all cases.
                total_assertions = 0
                passed_assertions = 0

                for r in summary.results:
                    if r.report:
                        result.total_cases = len(r.report.cases)
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
                                    result.failures.append(failure)

                result.passed_cases = passed_assertions
                result.total_cases = total_assertions
                result.pass_rate = (
                    passed_assertions / total_assertions
                    if total_assertions > 0
                    else 0.0
                )

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
