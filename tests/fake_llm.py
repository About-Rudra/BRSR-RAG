"""A scripted stand-in for a chat model, so the agent can be tested without an API key."""
import json
import re

from langchain_core.messages import AIMessage, ToolMessage


class FakeStructured:
    def __init__(self, parent, schema):
        self.parent, self.schema = parent, schema

    def invoke(self, prompt):
        self.parent.calls.append(("structured", self.schema.__name__))
        return self.parent.structured(self.schema, prompt)


class FakeChatModel:
    """respond(messages) -> AIMessage for tool-calling turns;
    structured(schema, prompt) -> schema instance for with_structured_output;
    text(prompt) -> str for plain prompts (the writer)."""

    def __init__(self, respond=None, structured=None, text=None):
        self.respond = respond or (lambda messages: AIMessage(content='{"claims": [], "note": "none"}'))
        self.structured = structured or (lambda schema, prompt: None)
        self.text = text or (lambda prompt: "answer")
        self.calls = []

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, schema):
        return FakeStructured(self, schema)

    def invoke(self, messages):
        self.calls.append(("invoke", None))
        if isinstance(messages, str):
            return AIMessage(content=self.text(messages))
        return self.respond(messages)


def tool_turn(name, args, call_id):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def last_tool_sids(messages):
    for m in reversed(messages):
        if isinstance(m, ToolMessage):
            return re.findall(r"\[(S\d+)\]", m.content)
    return []


def final(claims, note=""):
    return AIMessage(content=json.dumps({"claims": claims, "note": note}))
