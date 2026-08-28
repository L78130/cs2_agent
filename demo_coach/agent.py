import argparse
import json
import sys

from demo_coach.config import get_settings
from demo_coach.tools import MatchContext, TOOL_SCHEMAS, build_context, dispatch

SYSTEM = """你是一名犀利、诚实的 CS2 教练,正在复盘一名玩家的比赛。
比赛摘要(已计算好,直接信任这些数字 —— 绝不要重新计算):
{summary}

规则:
- 始终用中文回复。
- 不要编造数字。需要摘要之外的数据时,调用工具获取。
- 能力边界:你看不到准星摆放、反应时间和逐帧走位;涉及这些机制时只能
  推断,必须标注"可能"/"大概率",绝不要写成已经观察到的事实。
- 你报告的任何数字都必须来自摘要或工具返回的数据;不要自己发明统计量
  (如 p 值、相关系数、置信区间),也不要对数据做新的比率计算。
- 每条观察按「是什么 / 为什么 / 怎么做」组织:
  · 是什么 = 数据 + 数值 + 基准;没有基准就不要评判好坏;
  · 为什么 = 可能的原因(标注为推断);
  · 怎么做 = 具体、可以数得清的行动;禁用"提高""加强""注意""多练"这类空话。
- 每个结论必须引用回合号(第 N 回合)或工具返回的数据;不要凭印象下结论。
- 回合工具返回的击杀带有游戏内置点位名称(callout,如 BombsiteA、
  Middle);分析站位和走位时引用这些点位。
- 提及高光时刻时,注明类型和回合号。
- 评价玩家时必须同时覆盖道具使用,不能只看枪法:摘要和计分板带有
  util_dmg(道具总伤害)、util_dmg_r(道具伤害/局)、flash_assists(闪光助攻)、
  flashes_thrown / nades_thrown(闪光/投掷物数量)。结合回合详情判断道具是否
  服务于进点、反清、拖时间和残局(例如:道具伤害为 0 且闪光助攻为 0 的步枪手,
  说明道具没有转化为团队收益);参考基准:职业选手道具伤害约 5-10/局,
  闪光助攻每半场约 1-3 次。道具判断同样遵守「是什么/为什么/怎么做」。
"""

VERIFY = """你是一名事实核查员。下面是教练回复的草稿,上面的对话记录中包含工具返回的真实比赛数据。
逐条检查草稿中涉及具体数字、回合、玩家的论断:
- 有工具数据支持的保留;
- 没有数据支持或与数据矛盾的,删除或改写为有数据支持的说法;
- 不要添加新的论断,保持中文和原有结构。
只输出修订后的回复全文,不要输出任何解释。

草稿:
{reply}"""


def _usage_of(resp) -> dict | None:
    """Extract token usage from a response/stream chunk, if present."""
    u = getattr(resp, "usage", None)
    if u is None:
        return None
    try:
        return {"prompt": int(u.prompt_tokens or 0),
                "completion": int(u.completion_tokens or 0),
                "total": int(u.total_tokens or 0)}
    except (TypeError, ValueError, AttributeError):
        return None


def _add_usage(acc: dict, u: dict | None) -> None:
    if u:
        for k in acc:
            acc[k] += u[k]


_ENC = None


def _estimate_tokens(text: str) -> int:
    """Local token estimate (cl100k_base via tiktoken; char heuristic without it).
    The kimi coding endpoint omits usage from responses, so this is often the
    only usage signal we have."""
    global _ENC
    if not text:
        return 0
    try:
        if _ENC is None:
            import tiktoken
            _ENC = tiktoken.get_encoding("cl100k_base")
        return len(_ENC.encode(text))
    except Exception:
        return max(1, len(text) // 2)


def _estimate_usage(messages: list[dict], reply: str) -> dict:
    prompt = 0
    for m in messages:
        prompt += _estimate_tokens(m.get("content") or "")
        for tc in m.get("tool_calls") or []:
            prompt += _estimate_tokens(json.dumps(tc, default=str))
    completion = _estimate_tokens(reply)
    return {"prompt": prompt, "completion": completion,
            "total": prompt + completion, "estimated": True}


class CoachAgent:
    def __init__(self, ctx: MatchContext, client=None):
        self.ctx = ctx
        if client is None:
            settings = get_settings()
            if not settings.api_key:
                raise RuntimeError("MOONSHOT_API_KEY not set")
            from openai import OpenAI
            client = OpenAI(api_key=settings.api_key,
                            base_url=settings.base_url)
            self.model = settings.model
        else:
            self.model = get_settings().model
        self.client = client

    def _system(self) -> str:
        return SYSTEM.format(summary=json.dumps(self.ctx.summary,
                                                ensure_ascii=False, default=str))

    def _verify(self, messages: list[dict], reply: str) -> tuple[str, dict | None]:
        """Grounding pass: one extra call that drops or rewrites claims in the
        reply that the tool data does not support. Falls back to the original
        reply on any failure. Returns (reply, usage)."""
        if not reply:
            return reply, None
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages
                + [{"role": "user", "content": VERIFY.format(reply=reply)}])
            out = (resp.choices[0].message.content or "").strip()
            return out or reply, _usage_of(resp)
        except Exception:
            return reply, None

    def chat(self, message: str, history: list[dict]) -> str:
        messages = ([{"role": "system", "content": self._system()}]
                    + history + [{"role": "user", "content": message}])
        for _ in range(8):  # bound the tool-call loop
            resp = self.client.chat.completions.create(
                model=self.model, messages=messages, tools=TOOL_SCHEMAS)
            msg = resp.choices[0].message
            if not msg.tool_calls:
                reply, _ = self._verify(messages, msg.content or "")
                return reply
            messages.append(msg.model_dump(exclude_none=True))
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                result = dispatch(self.ctx, tc.function.name, args)
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": result})
        return "(已达到工具调用上限 —— 请换一个更具体的问题)"

    def chat_stream(self, message: str, history: list[dict]):
        """Streaming variant of chat(): yields SSE-ready event dicts —
        {"type": "token", "text": ...} for reply deltas,
        {"type": "tool", "name": ...} when a tool is invoked,
        {"type": "done", "reply": ...} last."""
        messages = ([{"role": "system", "content": self._system()}]
                    + history + [{"role": "user", "content": message}])
        reply = ""
        usage = {"prompt": 0, "completion": 0, "total": 0}
        for _ in range(8):  # bound the tool-call loop
            stream = self.client.chat.completions.create(
                model=self.model, messages=messages, tools=TOOL_SCHEMAS,
                stream=True, stream_options={"include_usage": True})
            content_parts: list[str] = []
            tool_calls: dict[int, dict] = {}
            for chunk in stream:
                _add_usage(usage, _usage_of(chunk))  # final chunk carries usage
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta is None:
                    continue
                if delta.content:
                    content_parts.append(delta.content)
                    yield {"type": "token", "text": delta.content}
                for tc in delta.tool_calls or []:
                    slot = tool_calls.setdefault(
                        tc.index, {"id": None, "name": "", "arguments": ""})
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function:
                        slot["name"] += tc.function.name or ""
                        slot["arguments"] += tc.function.arguments or ""
            if not tool_calls:
                reply = "".join(content_parts)
                break
            ordered = [tool_calls[i] for i in sorted(tool_calls)]
            messages.append({
                "role": "assistant",
                "content": "".join(content_parts) or None,
                "tool_calls": [
                    {"id": s["id"], "type": "function",
                     "function": {"name": s["name"], "arguments": s["arguments"]}}
                    for s in ordered],
            })
            for s in ordered:
                yield {"type": "tool", "name": s["name"]}
                args = json.loads(s["arguments"] or "{}")
                result = dispatch(self.ctx, s["name"], args)
                messages.append({"role": "tool", "tool_call_id": s["id"],
                                 "content": result})
        else:
            reply = "(已达到工具调用上限 —— 请换一个更具体的问题)"
        if not reply.startswith("("):
            reply, u = self._verify(messages, reply)
            _add_usage(usage, u)
        if not usage["total"]:
            # provider didn't report usage -> local estimate (marked as such)
            usage = _estimate_usage(messages, reply)
        yield {"type": "done", "reply": reply,
               "usage": usage if usage["total"] else None}


def main() -> None:
    ap = argparse.ArgumentParser(description="Chat with a CS2 demo coach")
    ap.add_argument("demo", help="path to .dem file")
    args = ap.parse_args()
    print("Parsing demo (cached after first run)...")
    ctx = build_context(args.demo)
    print(f"Loaded {ctx.summary['map']}, {ctx.summary['rounds_played']} rounds. "
          "Type your question, 'quit' to exit.")
    history: list[dict] = []
    agent = CoachAgent(ctx)
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q in ("quit", "exit", ""):
            break
        try:
            reply = agent.chat(q, history)
        except Exception as e:
            print(f"Error: LLM API call failed: {e}")
            continue
        history += [{"role": "user", "content": q},
                    {"role": "assistant", "content": reply}]
        print(reply)


if __name__ == "__main__":
    sys.exit(main())
