"""Run with DEEPSEEK_API_KEY or KIMI_API_KEY set."""

import asyncio

from agent0_runtime import Agent, tool


@tool("add")
def add(a: float, b: float) -> str:
    """Add two numbers."""
    return str(a + b)


async def main():
    async with Agent.from_env(tools=["add"]) as agent:
        result = await agent.run("Use add to calculate 20 + 22.")
        print(result.text)


if __name__ == "__main__":
    asyncio.run(main())
