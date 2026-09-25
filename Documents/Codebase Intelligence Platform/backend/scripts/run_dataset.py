#!/usr/bin/env python
"""
Runs an evaluation dataset and reports the result.

    ../venv/bin/python scripts/run_dataset.py --local              # no Langfuse, prints a table
    ../venv/bin/python scripts/run_dataset.py                      # pull from Langfuse, link a run
    ../venv/bin/python scripts/run_dataset.py --mode answer        # real answers (costs LLM calls)
    ../venv/bin/python scripts/run_dataset.py --dataset graph-grounding

`--local` runs the same items, the same task and the same evaluators as a Langfuse run — it just
reads them from `app/observability/datasets.py` instead of the API and prints instead of pushing.
That is what makes this usable in CI, and what makes a regression reproducible without a service.

Exit code is **1 when any item fails**, so this can gate a commit.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.observability import datasets as ds  # noqa: E402
from app.observability import runner  # noqa: E402
from app.observability import tracing  # noqa: E402


def _print_summary(title: str, summary: dict) -> None:
    print(f"\n{title}")
    print(f"  {summary['passed']}/{summary['items']} passed"
          + (f", {summary['errors']} errored" if summary["errors"] else ""))

    if summary["metrics"]:
        print("\n  metric                      mean")
        print("  " + "-" * 34)
        for name, value in summary["metrics"].items():
            print(f"  {name:<26} {value:>5.2f}")

    if summary["failures"]:
        print("\n  failures")
        for failure in summary["failures"]:
            print(f"    [{failure['check']}] {failure['query']}")
            if failure["why"]:
                print(f"        {failure['why']}")


def run_local(dataset_names, mode: str) -> int:
    """Executes the datasets from their definitions, with nothing to push to."""
    failed = 0
    # One fixture for the whole run: indexing it is the expensive part and every graph item asks
    # about the same repository.
    with runner.GraphFixture() as fixture:
        for name in dataset_names:
            items = ds.DATASETS[name]["items"]()
            results = runner.run_items(items, mode=mode, fixture=fixture)
            summary = runner.summarise(results)
            _print_summary(f"{name}  (mode: {mode})", summary)
            failed += summary["failed"] + summary["errors"]

    print()
    return 1 if failed else 0


def run_against_langfuse(dataset_names, mode: str, run_name: str, concurrency: int) -> int:
    """
    Pulls items from Langfuse and links the results to a named dataset run.

    The task and evaluators are the same functions `--local` uses; only the source of the items
    and the destination of the scores differ. Two code paths that scored differently would make
    a local pass and a dashboard failure impossible to reconcile.
    """
    from langfuse.experiment import Evaluation

    client = tracing._resolve_client()
    if client is None:
        print(
            "Langfuse is not configured, so there is no dataset to pull and nowhere to link a\n"
            "run. Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY, or pass --local.",
            file=sys.stderr,
        )
        return 1

    failed = 0
    with runner.GraphFixture() as fixture:
        for name in dataset_names:
            dataset = client.get_dataset(name)

            # Async because `answer` mode awaits the orchestrator; the SDK accepts either.
            async def task(*, item, **_kwargs):
                return await runner.aexecute(item.input, item.metadata or {},
                                             mode=mode, fixture=fixture)

            def evaluator(*, input, output, expected_output, metadata, **_kwargs):
                return [
                    Evaluation(name=e["name"], value=e["value"], comment=e["comment"])
                    for e in runner.evaluate(output, expected_output or {}, metadata or {})
                ]

            result = dataset.run_experiment(
                name=run_name,
                # Passed explicitly: `name` alone makes Langfuse append an ISO timestamp, so
                # runs could never be compared by a name you chose.
                run_name=run_name,
                description=f"{name} via scripts/run_dataset.py (mode: {mode})",
                task=task,
                evaluators=[evaluator],
                # The SDK default is 50. Against a local single-node Langfuse that floods
                # ingestion and items fail to link with "timed out" — while the run itself
                # reports success, because the work was done and only the recording was lost.
                max_concurrency=concurrency,
                metadata={"mode": mode},
            )

            # Summarised locally from the same evaluations, so the console output and the
            # dashboard cannot disagree about whether the run passed.
            results = [
                {
                    "id": r.item.id,
                    "query": (r.item.input or {}).get("query", ""),
                    "check": (r.item.metadata or {}).get("check"),
                    "evaluations": [{"name": e.name, "value": e.value, "comment": e.comment}
                                    for e in (r.evaluations or [])],
                    "passed": all(e.value == 1.0 for e in (r.evaluations or [])) or None,
                }
                for r in result.item_results
            ]
            for record in results:
                record["passed"] = (
                    all(e["value"] == 1.0 for e in record["evaluations"])
                    if record["evaluations"] else None
                )

            summary = runner.summarise(results)
            _print_summary(f"{name}  (run: {run_name}, mode: {mode})", summary)
            failed += summary["failed"] + summary["errors"]

    tracing.flush()
    print(f"\nLinked to dataset runs named '{run_name}'. "
          f"Compare at {os.getenv('LANGFUSE_HOST', 'http://localhost:3001')}/datasets")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--local", action="store_true",
                        help="run without Langfuse and print the result")
    parser.add_argument("--mode", choices=["facts", "answer"], default="facts",
                        help="'facts' is deterministic and free; 'answer' calls the LLM")
    parser.add_argument("--dataset", action="append", choices=sorted(ds.DATASETS),
                        help="limit to one dataset (repeatable)")
    parser.add_argument("--run-name", default=None,
                        help="name for the dataset run; defaults to a timestamp")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="parallel items when linking to Langfuse (default 4)")
    args = parser.parse_args()

    names = args.dataset or sorted(ds.DATASETS)
    if args.local:
        return run_local(names, args.mode)

    from datetime import datetime
    run_name = args.run_name or f"{args.mode}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    return run_against_langfuse(names, args.mode, run_name, args.concurrency)


if __name__ == "__main__":
    raise SystemExit(main())
