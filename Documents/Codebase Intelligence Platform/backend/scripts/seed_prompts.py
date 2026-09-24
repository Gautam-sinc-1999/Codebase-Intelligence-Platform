#!/usr/bin/env python
"""
Pushes the in-code prompts into Langfuse so they can be versioned there.

    ../venv/bin/python scripts/seed_prompts.py --dry-run     # show what would be sent
    ../venv/bin/python scripts/seed_prompts.py               # create a version, unlabelled
    ../venv/bin/python scripts/seed_prompts.py --promote     # ...and label it `production`
    ../venv/bin/python scripts/seed_prompts.py --status      # what is live right now

**Every run creates a new version.** That is how Langfuse works — versions are immutable and
auto-increment, and which one is live is decided by the `production` label, not by the number.
So seeding twice does not overwrite anything; it leaves an unused version behind.

`--promote` is separate from seeding on purpose. Creating a version is harmless; pointing live
traffic at it is a deployment, and the two should not be the same keystroke. Without it, the new
version sits unlabelled until you promote it — in the UI, or with `--promote-version N`.

Rolling back is the same operation in reverse: `--promote-version 3` moves `production` back to
v3, and the running process picks it up within LANGFUSE_PROMPT_CACHE_TTL. No redeploy.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings  # noqa: E402
from app.observability import prompts  # noqa: E402
from app.observability import tracing  # noqa: E402


def dry_run() -> int:
    for name, spec in prompts.REGISTRY.items():
        text = spec["text"]
        print(f"\n{name}  ({len(text)} chars, labels={spec['labels']})")
        print(f"  {spec['commit_message']}")
        print("  " + "-" * 70)
        for line in text.splitlines()[:6]:
            print(f"  {line[:74]}")
        if len(text.splitlines()) > 6:
            print(f"  … {len(text.splitlines()) - 6} more line(s)")
    print("\nNothing was sent.")
    return 0


def status() -> int:
    """What the running pipeline would use right now, and where it came from."""
    print(f"\nprompts enabled: {prompts.prompts_enabled()}"
          f"   cache ttl: {settings.LANGFUSE_PROMPT_CACHE_TTL}s")
    for name in prompts.REGISTRY:
        resolved = prompts.get(name)
        version = f"v{resolved.version}" if resolved.version else "—"
        labels = ",".join(resolved.labels) or "—"
        print(f"  {name:<24} {resolved.source:<10} {version:<5} labels={labels:<20} "
              f"{len(resolved.text)} chars")
    return 0


def seed(promote: bool) -> int:
    client = tracing._resolve_client()
    if client is None:
        print(
            "Langfuse is not configured, so there is nowhere to push prompts.\n"
            "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY, or pass --dry-run.",
            file=sys.stderr,
        )
        return 1

    for name, spec in prompts.REGISTRY.items():
        created = client.create_prompt(
            name=name,
            prompt=spec["text"],
            type="text",
            # Labelled only when explicitly promoting: creating a version is not a deployment.
            labels=spec["labels"] if promote else [],
            commit_message=spec["commit_message"],
        )
        version = getattr(created, "version", "?")
        where = "labelled production" if promote else "unlabelled"
        print(f"{name}: created v{version} ({where})")

    tracing.flush()
    if not promote:
        print("\nNothing is live yet. Promote a version in the UI, or re-run with --promote.")
    print(f"Review at {os.getenv('LANGFUSE_HOST', 'http://localhost:3001')}/prompts")
    return 0


def promote_version(name: str, version: int) -> int:
    """Moves the `production` label. This is the deploy, and the rollback."""
    client = tracing._resolve_client()
    if client is None:
        print("Langfuse is not configured.", file=sys.stderr)
        return 1

    client.update_prompt(name=name, version=version, new_labels=["production"])
    # Labels are unique across versions, so this removed the label from whichever version held
    # it. The SDK cache for this name is invalidated by the call; clear ours too so `--status`
    # immediately below reports the new state rather than the old one.
    client.clear_prompt_cache()
    print(f"{name}: 'production' now points at v{version}")
    print(f"Running processes pick this up within {settings.LANGFUSE_PROMPT_CACHE_TTL}s.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="print the prompts, send nothing")
    parser.add_argument("--status", action="store_true",
                        help="show which version the pipeline would use right now")
    parser.add_argument("--promote", action="store_true",
                        help="label the newly created versions `production`")
    parser.add_argument("--promote-version", type=int, metavar="N",
                        help="point `production` at an existing version (this is the rollback)")
    parser.add_argument("--name", choices=sorted(prompts.REGISTRY),
                        help="which prompt --promote-version applies to")
    args = parser.parse_args()

    if args.dry_run:
        return dry_run()
    if args.status:
        return status()
    if args.promote_version is not None:
        if not args.name:
            print("--promote-version needs --name to say which prompt.", file=sys.stderr)
            return 1
        return promote_version(args.name, args.promote_version)
    return seed(args.promote)


if __name__ == "__main__":
    raise SystemExit(main())
