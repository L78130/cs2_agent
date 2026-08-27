import json
from unittest.mock import MagicMock
from demo_coach import agent


def _msg(content=None, tool_calls=None):
    m = MagicMock()
    m.content = content
    m.tool_calls = tool_calls
    m.model_dump.return_value = {
        "role": "assistant", "content": content,
        **({"tool_calls": tool_calls} if tool_calls else {}),
    }
    return m


def _response(message):
    r = MagicMock()
    r.choices = [MagicMock(message=message)]
    return r


def test_chat_tool_loop():
    ctx = MagicMock()
    client = MagicMock()
    tc = MagicMock()
    tc.id = "call_1"
    tc.function.name = "get_scoreboard"
    tc.function.arguments = "{}"
    tc.model_dump.return_value = {"id": "call_1", "type": "function",
                                  "function": {"name": "get_scoreboard",
                                               "arguments": "{}"}}
    client.chat.completions.create.side_effect = [
        _response(_msg(tool_calls=[tc])),
        _response(_msg(content="bob top-fragged.")),
    ]
    a = agent.CoachAgent(ctx, client=client)
    reply = a.chat("who played best?", history=[])
    assert reply == "bob top-fragged."
    assert client.chat.completions.create.call_count == 2


def test_chat_plain_answer():
    ctx = MagicMock()
    client = MagicMock()
    client.chat.completions.create.return_value = _response(_msg(content="hi"))
    a = agent.CoachAgent(ctx, client=client)
    assert a.chat("hello", history=[]) == "hi"
