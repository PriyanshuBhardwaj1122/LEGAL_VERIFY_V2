"""Isolate the OpenAI structured-output call used by the planner, with full traceback."""

import asyncio
import traceback

from app.graph.nodes.planner import LLMResearchPlan
from app.providers.llm.base import get_llm


async def main():
    llm = get_llm()
    print(f"Using provider: {type(llm).__name__}\n")

    system = "You are a research director. Produce a research plan."
    user = (
        "Topic: Eligibility criteria under Section 29A of the Insolvency and "
        "Bankruptcy Code, 2016\n"
        "Produce a research plan with 3-6 legal issues and 4-8 search queries."
    )

    try:
        plan, usage = await llm.generate(
            system=system,
            user=user,
            output_schema=LLMResearchPlan,
            tool_name="emit_research_plan",
            max_tokens=4096,
            temperature=0.0,
        )
        print("SUCCESS")
        print(plan)
        print(usage)
    except Exception:
        print("FAILED — full traceback:\n")
        traceback.print_exc()


asyncio.run(main())