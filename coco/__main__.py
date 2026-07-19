import argparse
import asyncio
import sys

from dotenv import load_dotenv

load_dotenv()

from .agent import run_session
from .auth import AuthError


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="coco")
    parser.add_argument(
        "--admin",
        action="store_true",
        help=(
            "Start in local admin mode: bypass SSO and synthesize a "
            "full-trust admin identity. Gated by config.auth.allow_cli_admin "
            "(default false) — refuses when disallowed. Developer escape "
            "hatch for local iteration; never enable in production."
        ),
    )
    parser.add_argument(
        "--admin-name",
        default=None,
        help=(
            "Optional display name for the synthetic admin identity "
            "(defaults to 'local-admin'). Ignored unless --admin is set."
        ),
    )
    return parser.parse_args(argv)


def main():
    cli_flags = _parse_args()

    # --admin-name without --admin is a no-op; warn so the user notices.
    if cli_flags.admin_name and not cli_flags.admin:
        print(
            "[warning: --admin-name has no effect without --admin]",
            file=sys.stderr,
        )

    try:
        asyncio.run(run_session(cli_flags=cli_flags))
    except AuthError as e:
        print(f"[startup error: {e}]", file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
