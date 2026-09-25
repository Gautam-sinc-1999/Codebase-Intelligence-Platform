#!/usr/bin/env python
"""
Runs the RAGAS-style metrics over a dataset and pushes the results back as scores.

    ../venv/bin/python scripts/run_ragas.py --dry-run          # show the plan and the cost
    ../venv/bin/python scripts/run_ragas.py                    # measure, print, push
    ../venv/bin/python scripts/run_ragas.py --only faithfulness --limit 1

This is the offline half of the evaluation story. It never runs in the request path: the metrics
are LLM-as-judge and cost 15-30 calls per item, which is why they use `JUDGE_*` — a separate
provider from the one that answers questions, so evaluation never competes with a user waiting.

Needs two things configured:

  * an **answering** provider (`GROQ_API_KEY`), or every answer is the rule-based template
  * a **judging** provider (`JUDGE_API_KEY`), or there is nothing to measure with

Scores are attached to each item's own trace, so a low `faithfulness` is one click from the
retrieval and the prompt that produced it.
"""
import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx  # noqa: E402

from app.observability import datasets as ds  # noqa: E402
from app.observability import judge as judge_module  # noqa: E402
from app.observability import ragas_metrics as rm  # noqa: E402
from app.observability import runner  # noqa: E402
from app.observability import tracing  # noqa: E402

# Only items that produce an answer can be judged on one. The intent dataset classifies a query
# and returns no prose, so there is nothing for faithfulness or relevancy to read.
ANSWERABLE_CHECKS = {"graph_recall"}


def _contexts_from(output: dict) -> list:
    """The chunks the model was actually shown — what 'supported by the context' has to mean."""
    contexts = []
    for source in output.get("sources", []) or []:
        snippet = source.get("code_snippet") or source.get("code") or ""
        if snippet:
            label = f"{source.get('file_path', '')}:{source.get('start_line', '')}"
            contexts.append(f"{label}\n{snippet}")
    # The authoritative graph facts are part of the prompt too, and noise sensitivity exists
    # precisely to catch the model preferring a snippet over them.
    if output.get("facts"):
        contexts.append(output["facts"])
    return contexts


def _answerable(dataset_names):
    for name in dataset_names:
        for item in ds.DATASETS[name]["items"]():
            if item.get("metadata", {}).get("check") in ANSWERABLE_CHECKS:
                yield name, item


async def evaluate(dataset_names, only, limit, push) -> int:
    items = list(_answerable(dataset_names))
    if limit:
        items = items[:limit]

    if not items:
        print("No answerable items in the selected datasets.", file=sys.stderr)
        return 1

    metrics = only or rm.METRIC_NAMES
    judge = judge_module.Judge()
    totals = {}
    rows = []

    async with httpx.AsyncClient() as client:
        with runner.GraphFixture() as fixture:
            for dataset_name, item in items:
                question = item["input"]["query"]
                expected = item.get("expected_output", {})

                # One trace per item, so the scores below have somewhere to land and the answer's
                # own spans sit underneath them.
                async with tracing.start_trace(
                    "ragas_eval",
                    metadata={"dataset": dataset_name, "item_id": item["id"],
                              "judge": judge_module.describe_judge()},
                    tags=["ragas", dataset_name],
                ):
                    tracing.set_trace_io(input=question)
                    output = await runner.aexecute(item["input"], item["metadata"],
                                                   mode="answer", fixture=fixture)
                    tracing.set_trace_io(output=output.get("answer", ""))
                    trace_id = tracing.current_trace_id()

                    contexts = _contexts_from(output)
                    measured = await rm.evaluate_answer(
                        judge, client,
                        question=question,
                        answer=output.get("answer", ""),
                        contexts=contexts,
                        expected_entities=expected.get("callers", []),
                        only=metrics,
                    )

                    # The deterministic evaluations belong on the same trace: "the graph found all
                    # thirteen but faithfulness is 0.4" is a different diagnosis from both halves
                    # being low, and only having them together tells them apart.
                    for evaluation in runner.evaluate(output, expected, item["metadata"]):
                        measured.setdefault(evaluation["name"], {
                            "value": evaluation["value"], "comment": evaluation["comment"],
                        })

                    # Pushed **once**, on the open trace. An earlier version also re-sent every
                    # score by trace id as a belt-and-braces measure; both calls succeeded, so
                    # each metric landed twice and every dashboard average was computed over
                    # doubled rows — a silent 2x that looks like real data.
                    if push:
                        for name, result in measured.items():
                            tracing.score(name, result["value"],
                                          comment=result.get("comment"), data_type="NUMERIC")

                rows.append((question, output.get("degraded"), measured))
                for name, result in measured.items():
                    totals.setdefault(name, []).append(float(result["value"]))

    _report(rows, totals, judge)
    tracing.flush()
    return 0


def _report(rows, totals, judge) -> None:
    print(f"\njudge: {judge_module.describe_judge()}  ({judge.calls} calls)")
    for question, degraded, measured in rows:
        print(f"\n  {question}" + ("   [degraded answer — no LLM was reachable]" if degraded else ""))
        for name in sorted(measured):
            result = measured[name]
            arrow = "lower is better" if name == "noise_sensitivity" else ""
            print(f"    {name:<24} {float(result['value']):>5.2f}  {arrow}")
            if result.get("comment"):
                print(f"        {str(result['comment'])[:110]}")

    if totals:
        print("\n  mean across items")
        for name in sorted(totals):
            values = totals[name]
            print(f"    {name:<24} {sum(values) / len(values):>5.2f}")

    missing = set(rm.METRIC_NAMES) - set(totals)
    if missing:
        # Said out loud rather than left as a gap: a metric silently absent looks like a metric
        # that passed.
        print(f"\n  not measured: {', '.join(sorted(missing))}")


def dry_run(dataset_names, only, limit) -> int:
    items = list(_answerable(dataset_names))
    if limit:
        items = items[:limit]
    metrics = only or rm.METRIC_NAMES

    # Four of the five consult the judge; entity recall is an exact string search over symbol
    # names the dataset already states, so it costs nothing.
    judged = [m for m in metrics if m != "context_entity_recall"]
    print(f"\n{len(items)} answerable item(s), {len(metrics)} metric(s): {', '.join(metrics)}")
    for _, item in items:
        print(f"  - {item['input']['query']}")
    print(f"\nEach item costs 1 answering call plus ~{len(judged)} judging calls.")
    print(f"Roughly {len(items)} answering and {len(items) * len(judged)} judging calls in total.")
    print(f"judge: {judge_module.describe_judge()}")
    print("\nNothing was run.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--dataset", action="append", choices=sorted(ds.DATASETS))
    parser.add_argument("--only", action="append", choices=rm.METRIC_NAMES,
                        help="run only these metrics (repeatable)")
    parser.add_argument("--limit", type=int, default=None,
                        help="evaluate at most this many items")
    parser.add_argument("--dry-run", action="store_true",
                        help="show what would run, and what it would cost")
    parser.add_argument("--no-push", action="store_true",
                        help="measure and print without sending scores to Langfuse")
    args = parser.parse_args()

    names = args.dataset or sorted(ds.DATASETS)
    if args.dry_run:
        return dry_run(names, args.only, args.limit)

    if not judge_module.judge_configured():
        print(
            "No judge is configured, so there is nothing to measure with.\n"
            "Set JUDGE_API_KEY (Google AI Studio gives one free, no card:\n"
            "  https://aistudio.google.com/apikey ), or pass --dry-run.",
            file=sys.stderr,
        )
        return 1

    return asyncio.run(evaluate(names, args.only, args.limit, not args.no_push))


if __name__ == "__main__":
    raise SystemExit(main())
