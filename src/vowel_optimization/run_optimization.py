#!/usr/bin/env python3
"""Run prompt optimization for vowel eval spec generation.

Usage:
    # Evaluate current EVAL_SPEC_CONTEXT against playground functions
    python -m vowel_optimization.run_optimization eval

    # Run GePa optimization loop
    python -m vowel_optimization.run_optimization optimize --max-calls 50

    # Evaluate with a saved optimized context
    python -m vowel_optimization.run_optimization eval --context-file optimized_context.txt

    # Compare current vs optimized
    python -m vowel_optimization.run_optimization compare --context-file optimized_context.txt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import logfire
from vowel.context import EVAL_SPEC_CONTEXT

from .adapter import MODEL, PROPOSER_MODEL, create_adapter
from .functions import FUNCTION_CASES
from .task import generate_and_score

import dotenv 

dotenv.load_dotenv()

# ── Logfire setup ───────────────────────────────────────────
if os.getenv("LOGFIRE_ENABLED", "true").lower() == "true":
    logfire.configure(
        service_name="vowel-optimization",
        send_to_logfire="if-token-present",
    )
    logfire.instrument_pydantic_ai()
    logfire.instrument_httpx(capture_all=True)


def run_evaluation(
    eval_spec_context: str = EVAL_SPEC_CONTEXT,
    model: str = MODEL,
    label: str = "current",
) -> float:
    """Evaluate an EVAL_SPEC_CONTEXT against all playground functions.

    Returns average pass rate.
    """
    with logfire.span(
        "evaluation_run", label=label, context_chars=len(eval_spec_context)
    ):
        print(f"\n{'=' * 60}")
        print(f"Evaluating [{label}] context ({len(eval_spec_context)} chars)")
        print(f"{'=' * 60}")

        total_score = 0.0
        total_cases = 0
        total_passed = 0
        all_failures: dict[str, int] = {}

        for func_case in FUNCTION_CASES:
            print(f"\n  {func_case.name}...", end=" ", flush=True)
            result = generate_and_score(
                func_case=func_case,
                eval_spec_context=eval_spec_context,
                model=model,
            )

            total_score += result.score
            total_cases += result.total_cases
            total_passed += result.passed_cases

            if result.error:
                print(f"ERROR: {result.error[:80]}")
            else:
                print(
                    f"{result.pass_rate:.0%} ({result.passed_cases}/{result.total_cases})"
                )
                for f in result.failures:
                    all_failures[f.category] = all_failures.get(f.category, 0) + 1

        avg_score = total_score / len(FUNCTION_CASES) if FUNCTION_CASES else 0.0

        logfire.info(
            "evaluation_complete",
            label=label,
            avg_score=avg_score,
            total_passed=total_passed,
            total_cases=total_cases,
            failure_categories=all_failures,
        )

        print(f"\n{'-' * 60}")
        print(f"Average pass rate: {avg_score:.0%}")
        print(f"Total: {total_passed}/{total_cases} cases passed")

        if all_failures:
            print("\nFailure breakdown:")
            for cat, count in sorted(all_failures.items(), key=lambda x: -x[1]):
                print(f"  {cat}: {count}")

        return avg_score


def run_optimization(
    max_metric_calls: int = 50,
    output_file: str | None = None,
    eval_model: str = MODEL,
    proposer_model: str = PROPOSER_MODEL,
    seed_file: str | None = None,
) -> str:
    """Run GePa optimization to improve EVAL_SPEC_CONTEXT.

    Returns the optimized context string.
    """
    from gepa import optimize

    with logfire.span(
        "optimization_run",
        eval_model=eval_model,
        proposer_model=proposer_model,
        max_metric_calls=max_metric_calls,
        num_functions=len(FUNCTION_CASES),
    ):
        print("\nStarting GePa prompt optimization...")
        print(f"  Eval model: {eval_model}")
        print(f"  Proposer model: {proposer_model}")
        print(f"  Max metric calls: {max_metric_calls}")
        # Use seed file if provided, otherwise default context
        if seed_file:
            seed_context = Path(seed_file).read_text()
            print(f"  Seed: {seed_file} ({len(seed_context)} chars)")
        else:
            seed_context = EVAL_SPEC_CONTEXT
            print(f"  Seed: default EVAL_SPEC_CONTEXT ({len(seed_context)} chars)")
        print(f"{'=' * 60}")

        adapter = create_adapter(
            eval_model=eval_model,
            proposer_model=proposer_model,
        )

        seed_candidate = {
            "eval_spec_context": json.dumps(seed_context),
        }

        logfire.info("gepa_optimize_start", seed_context_chars=len(seed_context))

        result = optimize(
            seed_candidate=seed_candidate,
            trainset=FUNCTION_CASES,
            valset=FUNCTION_CASES,  # same set for now (small dataset)
            adapter=adapter,
            max_metric_calls=max_metric_calls,
            display_progress_bar=True,
        )

        best_context = json.loads(result.best_candidate["eval_spec_context"])
        best_score = result.val_aggregate_scores[result.best_idx]

        logfire.info(
            "optimization_complete",
            best_score=best_score,
            best_context_chars=len(best_context),
            total_candidates=len(result.val_aggregate_scores),
        )

        print(f"\n{'=' * 60}")
        print("Optimization Complete!")
        print(f"{'=' * 60}")
        print(f"Best validation score: {best_score:.2%}")
        print(f"Context length: {len(best_context)} chars")

        if output_file:
            Path(output_file).write_text(best_context)
            print(f"Saved to: {output_file}")
        else:
            # Default save
            default_path = Path("vowel_optimization/optimized_context.txt")
            default_path.write_text(best_context)
            print(f"Saved to: {default_path}")

        return best_context


def run_compare(
    context_file: str,
    model: str = MODEL,
) -> None:
    """Compare current EVAL_SPEC_CONTEXT vs an optimized one."""
    optimized = Path(context_file).read_text()

    print("\n1. Current EVAL_SPEC_CONTEXT:")
    score_current = run_evaluation(EVAL_SPEC_CONTEXT, model=model, label="current")

    print("\n2. Optimized EVAL_SPEC_CONTEXT:")
    score_optimized = run_evaluation(optimized, model=model, label="optimized")

    print(f"\n{'=' * 60}")
    print("Comparison:")
    print(f"  Current:   {score_current:.0%}")
    print(f"  Optimized: {score_optimized:.0%}")
    diff = score_optimized - score_current
    print(f"  Δ: {diff:+.0%}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Vowel eval spec prompt optimization via GePa"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # eval
    eval_p = subparsers.add_parser("eval", help="Evaluate current context")
    eval_p.add_argument(
        "--context-file", type=str, help="Use a saved context file instead of default"
    )
    eval_p.add_argument("--model", type=str, default=MODEL, help="Eval model")

    # optimize
    opt_p = subparsers.add_parser("optimize", help="Run GePa optimization")
    opt_p.add_argument("--max-calls", type=int, default=50, help="Max metric calls")
    opt_p.add_argument("--output", type=str, help="File to save optimized context")
    opt_p.add_argument("--model", type=str, default=MODEL, help="Eval model")
    opt_p.add_argument(
        "--proposer-model", type=str, default=PROPOSER_MODEL, help="Proposer model"
    )
    opt_p.add_argument(
        "--seed-file", type=str, help="Start from a previously optimized context file"
    )

    # compare
    cmp_p = subparsers.add_parser("compare", help="Compare current vs optimized")
    cmp_p.add_argument("context_file", type=str, help="Path to optimized context file")
    cmp_p.add_argument("--model", type=str, default=MODEL, help="Eval model")

    args = parser.parse_args()

    if args.command == "eval":
        if args.context_file:
            ctx = Path(args.context_file).read_text()
            run_evaluation(ctx, model=args.model, label="from-file")
        else:
            run_evaluation(model=args.model)

    elif args.command == "optimize":
        run_optimization(
            max_metric_calls=args.max_calls,
            output_file=args.output,
            eval_model=args.model,
            proposer_model=args.proposer_model,
            seed_file=args.seed_file,
        )

    elif args.command == "compare":
        run_compare(args.context_file, model=args.model)

    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
