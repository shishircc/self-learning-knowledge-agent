import asyncio
import sys

from dotenv import load_dotenv

load_dotenv()

from .agent import run_session
from .auth import AuthError


def main():
    try:
        asyncio.run(run_session())
    except AuthError as e:
        print(f"[startup error: {e}]", file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
