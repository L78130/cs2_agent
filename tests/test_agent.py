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
        _response(_msg(content="bob top-fragged. (已核实)")),  # verification pass
    ]
    a = agent.CoachAgent(ctx, client=client)
    reply = a.chat("who played best?", history=[])
    assert reply == "bob top-fragged. (已核实)"
    assert client.chat.completions.create.call_count == 3


def test_chat_plain_answer():
    ctx = MagicMock()
    client = MagicMock()
    client.chat.completions.create.return_value = _response(_msg(content="hi"))
    a = agent.CoachAgent(ctx, client=client)
    assert a.chat("hello", history=[]) == "hi"


def _chunk(content=None, tc_parts=None, usage=None):
    """One streaming chunk; tc_parts = list of (index, id, name, arguments)."""
    delta = MagicMock()
    delta.content = content
    tcs = []
    for index, tc_id, name, arguments in (tc_parts or []):
        tc = MagicMock()
        tc.index = index
        tc.id = tc_id
        tc.function.name = name
        tc.function.arguments = arguments
        tcs.append(tc)
    delta.tool_calls = tcs
    chunk = MagicMock()
    chunk.choices = [MagicMock(delta=delta)]
    chunk.usage = usage
    return chunk


def test_chat_stream_tool_loop():
    ctx = MagicMock()
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        iter([_chunk(tc_parts=[(0, "call_1", "get_sc", None)]),
              _chunk(tc_parts=[(0, None, "oreboard", "{")]),
              _chunk(tc_parts=[(0, None, None, "}")])]),
        iter([_chunk("bob "), _chunk("top-fragged.")]),
    ]
    a = agent.CoachAgent(ctx, client=client)
    events = list(a.chat_stream("who played best?", history=[]))
    assert [e["type"] for e in events] == ["tool", "token", "token", "done"]
    assert events[0]["name"] == "get_scoreboard"
    assert events[1]["text"] == "bob " and events[2]["text"] == "top-fragged."
    assert events[3]["reply"] == "bob top-fragged."
    # tool result was dispatched with the reassembled arguments
    tool_msgs = [m for c in client.chat.completions.create.call_args_list
                 for m in c.kwargs["messages"] if m.get("role") == "tool"]
    assert tool_msgs and tool_msgs[0]["tool_call_id"] == "call_1"


def test_chat_stream_plain_answer():
    ctx = MagicMock()
    client = MagicMock()
    client.chat.completions.create.return_value = iter([_chunk("hi")])
    a = agent.CoachAgent(ctx, client=client)
    events = list(a.chat_stream("hello", history=[]))
    assert events[0] == {"type": "token", "text": "hi"}
    assert events[1]["type"] == "done" and events[1]["reply"] == "hi"
    # provider reported no usage -> local estimate kicks in
    assert events[1]["usage"]["estimated"] is True
    assert events[1]["usage"]["total"] > 0


def _usage(p, c):
    return MagicMock(prompt_tokens=p, completion_tokens=c, total_tokens=p + c)


def test_chat_stream_token_usage():
    # usage from the final stream chunk + the verification call are summed
    ctx = MagicMock()
    client = MagicMock()
    final = MagicMock()          # usage-only trailing chunk (no choices)
    final.choices = []
    final.usage = _usage(10, 5)
    vresp = MagicMock()          # verification call (non-streaming)
    vresp.choices = [MagicMock(message=MagicMock(content="hi"))]
    vresp.usage = _usage(3, 1)
    client.chat.completions.create.side_effect = [
        iter([_chunk("hi"), final]), vresp]
    a = agent.CoachAgent(ctx, client=client)
    events = list(a.chat_stream("hello", history=[]))
    assert events[-1]["usage"] == {"prompt": 13, "completion": 6, "total": 19}
