#!/usr/bin/env python
"""
Pushes the evaluation datasets into Langfuse.

    ../venv/bin/python scripts/seed_datasets.py            # seed
    ../venv/bin/python scripts/seed_datasets.py --dry-run  # show what would be sent

Idempotent. Every item carries a stable id derived from its input, so running this twice updates
the existing items rather than doubling the dataset — a seeder that silently doubles its dataset
changes what every later run measures, and the numbers drift without anything looking broken.

The data comes from `app/observability/datasets.py`, which is also what `tests/test_intent.py` and
`tests/test_graph_facts.py` assert against. Seeding therefore cannot disagree with the suite.

Requires LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY unless --dry-run is given.
"""
import argparse
import os
import sys

# Runnable from anywhere: the script lives beside the package it imports.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.observability import datasets as ds  # noqa: E402
from app.observability import tracing  # noqa: E402


def _plan():
    """The datasets and their items, resolved once so dry-run and seed cannot diverge."""
    return [
        (name, spec["description"], spec["items"]())
        for name, spec in ds.DATASETS.items()
    ]


def dry_run() -> int:
    total = 0
    for name, description, items in _plan():
        print(f"\n{name}  ({len(items)} items)")
        print(f"  {description}")
        for item in items:
            check = item["metadata"].get("check", "?")
            expected = item["expected_output"]
            print(f"    [{check:<18}] {item['input']['query'][:58]:<58} -> {expected}")
        total += len(items)
    print(f"\n{total} items across {len(ds.DATASETS)} datasets. Nothing was sent.")
    return 0


def seed() -> int:
    client = tracing._resolve_client()
    if client is None:
        # Not an error worth a stack trace: the usual cause is simply not having configured
        # Langfuse yet, and the fix is two environment variables.
        print(
            "Langfuse is not configured, so there is nowhere to seed.\n"
            "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY (see the README's Observability\n"
            "section), or pass --dry-run to see what would be sent.",
            file=sys.stderr,
        )
        return 1

    created = 0
    for name, description, items in _plan():
        client.create_dataset(name=name, description=description)
        print(f"{name}: {len(items)} items")
        for item in items:
            client.create_dataset_item(
                dataset_name=name,
                id=item["id"],
                input=item["input"],
                expected_output=item["expected_output"],
                metadata=item["metadata"],
            )
            created += 1

    # Datasets are queued like everything else in the SDK; a script that exits without flushing
    # sends nothing at all.
    tracing.flush()
    print(f"\nSeeded {created} items. "
          f"Open {os.getenv('LANGFUSE_HOST', 'http://localhost:3001')}/datasets to review them.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--dry-run", action="store_true",
                        help="print the items instead of sending them")
    args = parser.parse_args()
    return dry_run() if args.dry_run else seed()


if __name__ == "__main__":
    raise SystemExit(main())
