import asyncio
from .agent import run_session
from dotenv import load_dotenv

load_dotenv()

def main():
    asyncio.run(run_session())


if __name__ == "__main__":
    main()
