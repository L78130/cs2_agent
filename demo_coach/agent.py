import argparse
import json
import sys

from demo_coach.config import get_settings
from demo_coach.tools import MatchContext, TOOL_SCHEMAS, build_context, dispatch

SYSTEM = """You are a sharp, honest CS2 coach reviewing one of the player's matches.
Match summary (already computed, trust these numbers — never recompute them):
{summary}

Rules:
- Never invent numbers. If you need data not in the summary, call a tool.
- Coaching feedback: concrete, specific, tied to rounds. No generic advice.
- Highlights: reference type/round when mentioning them.
- Reply in the same language the user writes in.
"""


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

    def chat(self, message: str, history: list[dict]) -> str:
        messages = ([{"role": "system", "content": self._system()}]
                    + history + [{"role": "user", "content": message}])
        for _ in range(8):  # bound the tool-call loop
            resp = self.client.chat.completions.create(
                model=self.model, messages=messages, tools=TOOL_SCHEMAS)
            msg = resp.choices[0].message
            if not msg.tool_calls:
                return msg.content or ""
            messages.append(msg.model_dump(exclude_none=True))
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                result = dispatch(self.ctx, tc.function.name, args)
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": result})
        return "(tool-call limit reached — please ask a narrower question)"


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
        reply = agent.chat(q, history)
        history += [{"role": "user", "content": q},
                    {"role": "assistant", "content": reply}]
        print(reply)


if __name__ == "__main__":
    sys.exit(main())
