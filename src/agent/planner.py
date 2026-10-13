"""Planner node: splits the question into 1-4 sub-questions, each tagged with a regime."""
from typing import List, Literal

from pydantic import BaseModel, Field

from src.rag import llm as llm_module
from src.agent import prompts
from src.agent.state import MAX_SUBQUESTIONS, event


class SubQuestionModel(BaseModel):
    text: str = Field(description="A self-contained sub-question.")
    regime: Literal["sebi_brsr", "cpcb_epr", "ccts", "any"]


class PlanModel(BaseModel):
    subquestions: List[SubQuestionModel] = Field(min_length=1, max_length=MAX_SUBQUESTIONS)


def make_plan(question):
    """Returns (plan, llm calls used, trace events). Never raises."""
    model = llm_module.get_chat_model().with_structured_output(PlanModel)
    prompt = prompts.PLANNER_PROMPT.format(regime_guide=prompts.REGIME_GUIDE, question=question)
    trace, llm_calls = [], 0

    for attempt in (1, 2):  # one retry on invalid output
        llm_calls += 1
        try:
            result = model.invoke(prompt)
            if isinstance(result, dict):
                result = PlanModel.model_validate(result)
            if result is not None:
                plan = [{"id": f"Q{i}", "text": sq.text.strip(), "regime": sq.regime}
                        for i, sq in enumerate(result.subquestions, start=1)]
                trace.append(event("planner", "plan", attempt=attempt, plan=plan))
                return plan, llm_calls, trace
            error = "empty output"
        except Exception as e:
            error = str(e)
        trace.append(event("planner", "invalid_output", attempt=attempt, error=error[:300]))

    plan = [{"id": "Q1", "text": question, "regime": "any"}]
    trace.append(event("planner", "fallback", plan=plan))
    return plan, llm_calls, trace


def planner_node(state):
    plan, llm_calls, trace = make_plan(state["question"])
    return {"plan": plan, "to_research": [s["id"] for s in plan],
            "trace": state["trace"] + trace, "llm_calls": state["llm_calls"] + llm_calls}
