"""GEPA adapter for vowel eval spec prompt optimization.

Bridges GEPA's optimize() loop with vowel's eval generation:
- evaluate: generate eval specs + run against original functions → score
- make_reflective_dataset: build structured failure feedback for the proposer
- propose_new_texts: use a proposer LLM to suggest improved EVAL_SPEC_CONTEXT
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import logfire
from gepa.core.adapter import EvaluationBatch, GEPAAdapter
from pydantic_ai import Agent

from .functions import FunctionCase
from .task import EvalResult, generate_and_score

MODEL = "openrouter:google/gemini-3-flash-preview"
PROPOSER_MODEL = "openrouter:google/gemini-3-flash-preview"


@dataclass
class VowelTrajectory:
    """Trajectory data from running eval generation for one function."""

    func_case: FunctionCase
    eval_result: EvalResult


@dataclass
class VowelGEPAAdapter(
    GEPAAdapter[
        FunctionCase,  # BatchItemT
        VowelTrajectory,  # TrajectoryT
        dict | None,  # OutputT
    ]
):
    """GEPA adapter for optimizing vowel's EVAL_SPEC_CONTEXT prompt.

    The candidate dict has key "eval_spec_context" containing the prompt.
    Each batch item is a FunctionCase (name, code, func, description).
    Score = pass rate of generated evals run against the original function.
    """

    eval_model: str = MODEL
    proposer_model: str = PROPOSER_MODEL
    _proposer_agent: Agent[None, str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._proposer_agent = Agent(
            self.proposer_model,
            output_type=str,
            defer_model_check=True,
            instructions="""You are an expert at improving LLM prompts for code evaluation generation.

You will receive:
1. The current EVAL_SPEC_CONTEXT prompt used to guide eval YAML generation
2. Detailed failure feedback from running generated evals against ground-truth function implementations

Your task: propose an IMPROVED version of the EVAL_SPEC_CONTEXT that fixes the observed failures.

Focus on:
- If expected values are wrong → add stronger "trace the algorithm" guidelines
- If invented raises appear → add "only test raises for code paths that actually throw"
- If assertions are over-strict → add "prefer property-based assertions over exact checks"
- If format mismatches occur → add "check exact formatting details (separators, whitespace)"
- If type errors occur → add supported type expressions list

Return ONLY the improved EVAL_SPEC_CONTEXT text (no markdown fences, no explanation).""",
        )

        # Required by GEPA protocol
        self.propose_new_texts = self._propose_new_texts_impl

    def evaluate(
        self,
        batch: list[FunctionCase],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[VowelTrajectory, dict | None]:
        """Evaluate a candidate EVAL_SPEC_CONTEXT on a batch of functions.

        For each function: generate eval spec → run against original → score.
        """
        eval_spec_context = json.loads(candidate["eval_spec_context"])

        outputs: list[dict | None] = []
        scores: list[float] = []
        trajectories: list[VowelTrajectory] | None = [] if capture_traces else None

        for func_case in batch:
            result = generate_and_score(
                func_case=func_case,
                eval_spec_context=eval_spec_context,
                model=self.eval_model,
            )

            outputs.append(
                {
                    "func_name": result.func_name,
                    "pass_rate": result.pass_rate,
                    "total_cases": result.total_cases,
                    "passed_cases": result.passed_cases,
                    "error": result.error,
                }
            )
            scores.append(result.score)

            if capture_traces and trajectories is not None:
                trajectories.append(
                    VowelTrajectory(
                        func_case=func_case,
                        eval_result=result,
                    )
                )

            print(
                f"  {func_case.name}: {result.pass_rate:.0%} ({result.passed_cases}/{result.total_cases})"
            )

        avg = sum(scores) / len(scores) if scores else 0.0
        print(f"  → Average: {avg:.0%}")

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[VowelTrajectory, dict | None],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """Build structured failure feedback for the proposer LLM."""
        if eval_batch.trajectories is None:
            return {}

        examples: list[dict[str, Any]] = []

        for traj, score in zip(eval_batch.trajectories, eval_batch.scores, strict=False):
            result = traj.eval_result
            record: dict[str, Any] = {
                "function": result.func_name,
                "score": score,
                "total_cases": result.total_cases,
                "passed_cases": result.passed_cases,
                "feedback": result.feedback_text(),
            }

            # Add failure categorization
            if result.failures:
                categories: dict[str, int] = {}
                for f in result.failures:
                    categories[f.category] = categories.get(f.category, 0) + 1
                record["failure_categories"] = categories

                # Include up to 5 detailed failures
                record["sample_failures"] = [
                    {
                        "case_id": f.case_id,
                        "evaluator": f.evaluator,
                        "reason": f.reason,
                        "category": f.category,
                    }
                    for f in result.failures[:5]
                ]

            if result.error:
                record["error"] = result.error

            examples.append(record)

        return {"eval_spec_context": examples}

    def _propose_new_texts_impl(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        """Propose improved EVAL_SPEC_CONTEXT based on failure feedback."""
        with logfire.span("GEPA_propose_new_texts"):
            current_context = json.loads(candidate["eval_spec_context"])
            examples = reflective_dataset.get("eval_spec_context", [])

            if not examples:
                return candidate

            # Build feedback summary
            feedback_parts = []
            for ex in examples:
                feedback_parts.append(ex.get("feedback", ""))
                cats = ex.get("failure_categories", {})
                if cats:
                    feedback_parts.append(f"  Failure distribution: {json.dumps(cats)}")
                for sf in ex.get("sample_failures", []):
                    feedback_parts.append(
                        f"  [{sf['category']}] {sf['case_id']}: {sf['reason'][:200]}"
                    )

            feedback_text = "\n\n".join(feedback_parts)
            logfire.info(
                "proposer_feedback",
                num_examples=len(examples),
                feedback_chars=len(feedback_text),
            )

            prompt = f"""Current EVAL_SPEC_CONTEXT (the prompt being optimized):
--- START ---
{current_context}
--- END ---

Evaluation Results Against Ground-Truth Functions:
{feedback_text}

Based on this feedback, propose an improved EVAL_SPEC_CONTEXT.
The improved version should:
1. Fix the root causes of the observed failures
2. Keep all structural YAML documentation intact
3. Strengthen quality guidelines based on specific failure patterns
4. NOT remove any evaluator type documentation

Return ONLY the improved EVAL_SPEC_CONTEXT text."""

            result = self._proposer_agent.run_sync(prompt)
            new_context = result.output
            logfire.info("new_context_proposed", new_context_chars=len(new_context))

            return {"eval_spec_context": json.dumps(new_context)}


def create_adapter(
    eval_model: str = MODEL,
    proposer_model: str = PROPOSER_MODEL,
) -> VowelGEPAAdapter:
    """Create a VowelGEPAAdapter for prompt optimization."""
    return VowelGEPAAdapter(
        eval_model=eval_model,
        proposer_model=proposer_model,
    )
