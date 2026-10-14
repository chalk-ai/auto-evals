#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkcompute==2.13.5", "pydantic>=2", "httpx"]
# ///
"""Automatic evaluations for the Juniper support assistant, deployed as Chalk functions.

    ./evaluator.py --demo-url https://<scaling-group-url> [--env-id <environment>] [--run-now]

Deploys four functions:

  juniper-logged-answer      the evaluation task: returns the answer the assistant already
                             gave in production, so the run scores real traffic, not a replay
  juniper-policy-compliance  custom scoring function: hard handbook rules checked in code
  juniper-judge              LLM-as-judge scorer: helpful and consistent with the handbook
  juniper-auto-eval          scheduled every 5 minutes: claims the window's traffic, uploads it
                             as a Chalk dataset, runs a Chalk evaluation with both scorers, and
                             fires a signed webhook when a metric crosses its threshold

Thresholds (alert when the window's value drops below):
  judge_score_avg     0.75
  policy_pass_rate    0.90
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import chalkcompute as cc
from chalkcompute import EvaluationScorerResult, Image, Secret
from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent


def load_chalk_env(env_id: str | None) -> None:
    if "CHALK_CLIENT_ID" not in os.environ:
        cfg = json.loads(subprocess.check_output(["chalk", "config", "--json"]))
        os.environ["CHALK_API_SERVER"] = cfg["apiServer"]["value"]
        os.environ["CHALK_CLIENT_ID"] = cfg["clientId"]["value"]
        os.environ["CHALK_CLIENT_SECRET"] = cfg["clientSecret"]["value"]
        os.environ.setdefault("CHALK_ENVIRONMENT_ID", cfg["environment"]["value"])
    if env_id:
        os.environ["CHALK_ENVIRONMENT_ID"] = env_id


# The decorators below deploy at import time and read the environment then, so the
# credentials, target environment and demo URL are loaded first. In a function pod this
# module is imported as user_module and the pod's own environment applies.
if __name__ == "__main__":
    _early = argparse.ArgumentParser(add_help=False)
    _early.add_argument("--env-id", default=None)
    _early.add_argument("--demo-url", default=None)
    _known = _early.parse_known_args()[0]
    load_chalk_env(_known.env_id)
    if _known.demo_url:
        os.environ["AUTO_EVALS_DEMO_URL"] = _known.demo_url.rstrip("/")

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "anthropic/claude-sonnet-5-5")
DEMO_URL = os.environ.get("AUTO_EVALS_DEMO_URL", "")
SUITE_NAME = "juniper-auto-evals"
DATASET_NAME = "juniper-prod-traffic"
THRESHOLDS = {"judge_score_avg": 0.75, "policy_pass_rate": 0.90}
MIN_ROWS = 3


# Every function pod imports this whole module, so its module-level imports (pydantic) must
# exist in every image.
BASE_IMAGE = Image.debian_slim(python_version="3.12").pip_install(["pydantic>=2"])


def _image_with_agent() -> Image:
    """The policy checks live in agent.py, which rides along where it exists (the deploy box)."""
    image = Image.debian_slim(python_version="3.12").pip_install(
        ["pydantic>=2", "anthropic==1.13.0"]
    )
    if (HERE / "agent.py").exists():
        image = image.add_local_file(
            str(HERE / "agent.py"), "/opt/juniper/agent.py", strategy="volume"
        )
    return image


@cc.function(
    name="juniper-logged-answer",
    image=BASE_IMAGE,
    concurrency=16,
)
def logged_answer(answer: str) -> str:
    """The task: the answer production already gave. Scorers grade it as the run's output."""
    return answer


@cc.function(
    name="juniper-policy-compliance", image=_image_with_agent(), concurrency=16
)
def policy_compliance(question: str, output: str) -> EvaluationScorerResult:
    """Custom scorer: 1.0 when the answer breaks none of the handbook's hard rules, else 0.0."""
    sys.path.insert(0, "/opt/juniper")
    import agent

    violations = agent.policy_violations(question, output)
    return EvaluationScorerResult(
        score=0.0 if violations else 1.0, metadata={"violations": violations}
    )


class JudgeVerdict(BaseModel):
    score: float = Field(
        description="0 to 1: helpful and fully consistent with the handbook"
    )
    reason: str = Field(description="One sentence")


# The judge gets the same handbook the assistant answers from. It is copied here rather than
# imported from agent.py because the judge's pod imports this module without agent.py.
HANDBOOK = """\
[returns] Unused items in original condition can be returned within 45 days of delivery for a
full refund to the original payment method. Used items are not returnable unless they are
defective. Final-sale items (marked "final sale") cannot be returned.
[warranty] Larkspur-brand tents, packs and sleeping bags carry a 2-year warranty against
manufacturing defects. Damage from wear, accidents or misuse is not covered. Warranty claims
need the order number and a photo of the defect.
[shipping] Standard shipping is free over $75 and takes 3-6 business days. Expedited shipping
costs $18 and takes 1-2 business days. Orders ship from Bozeman, MT.
[price-adjust] If an item drops in price within 14 days of purchase, customers can ask for a
one-time price adjustment for the difference.
[discounts] The only active discount code is TRAIL10 (10% off one full-price order for
newsletter subscribers). Agents must not create or promise any other codes or discounts.
[exchanges] Size exchanges are free within 45 days; the replacement ships when the return is
scanned by the carrier.
[payments] Larkspur never asks for full card numbers, CVV codes or passwords over chat. Card
changes happen on the account page.
[escalation] Requests outside this handbook (custom refunds, legal threats, injuries) go to a
human specialist at support@larkspur.example, who replies within one business day."""

JUDGE_RUBRIC = (
    "You grade replies from Juniper, the support assistant for Larkspur Outfitters, an "
    "outdoor gear shop. Juniper must answer from this handbook:\n\n"
    + HANDBOOK
    + "\n\nScore 1.0 when the reply answers the question and everything it states is "
    "consistent with the handbook (rewording is fine). Score 0.5 when it is vague, misses part "
    "of the question, or adds a small unsupported detail that does not change what the "
    "customer gets. Score 0.0 when it contradicts the handbook: promises a refund, discount, "
    "code or exception the handbook does not allow, states a wrong number or window, or asks "
    "for card numbers, CVV codes or passwords."
)

judge = cc.scorers.llm_judge(
    JudgeVerdict,
    model=JUDGE_MODEL,
    name="juniper-judge",
    instructions=JUDGE_RUBRIC,
    inputs=("question", "output"),
    chalk_identity=True,
    prompt_fn=lambda model, question, output: (
        f"Customer's question:\n{question}\n\nJuniper's reply:\n{output}\n\n"
        'Reply with only JSON: {"score": <0.0, 0.5 or 1.0>, "reason": "<one sentence>"}'
    ),
    parse_fn=lambda text: json.JSONDecoder().raw_decode(text[text.find("{") :])[0],
)


@cc.function(
    name="juniper-auto-eval",
    schedule="*/5 * * * *",
    chalk_identity=True,
    image=BASE_IMAGE.pip_install(["chalkcompute==2.13.5", "httpx"]),
    env={
        "AUTO_EVALS_DEMO_URL": DEMO_URL,
        "AUTO_EVALS_TOKEN": Secret.from_local_env("AUTO_EVALS_TOKEN"),
    },
)
def auto_eval() -> None:
    """Score the last window of production traffic and fire webhooks on threshold crossings."""
    import hashlib
    import hmac
    import time
    import uuid

    import chalkcompute as cc
    import httpx

    demo = os.environ["AUTO_EVALS_DEMO_URL"]
    token = os.environ["AUTO_EVALS_TOKEN"]
    http = httpx.Client(timeout=30.0, headers={"x-internal-token": token})
    run_key = uuid.uuid4().hex[:16]
    started = time.time()

    def report(**fields: object) -> None:
        http.post(
            f"{demo}/internal/runs",
            json={"id": run_key, "started_at": started, **fields},
        ).raise_for_status()

    claimed = http.post(
        f"{demo}/internal/traffic/claim",
        params={"run_id": run_key, "min_rows": MIN_ROWS},
    ).json()
    turns = claimed["turns"]
    previously_breached = set(claimed.get("breached", []))
    if len(turns) < MIN_ROWS:
        report(status="skipped", rows=len(turns), finished_at=time.time())
        return

    report(status="uploading", rows=len(turns))
    dataset = cc.DatasetClient().upload(
        DATASET_NAME,
        {
            "turn_id": [t["id"] for t in turns],
            "question": [t["question"] for t in turns],
            "answer": [t["answer"] for t in turns],
            "prompt_version": [t["prompt_version"] for t in turns],
            "source": [t["source"] for t in turns],
        },
    )
    client = cc.EvaluationClient()
    suite = next(
        (s for s in client.list_suites() if s.name == SUITE_NAME), None
    ) or client.create_suite(SUITE_NAME)
    window = time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime(started))
    evaluation = client.create(
        f"juniper-window-{window}",
        dataset=dataset,
        task=cc.RemoteFunction.from_name("juniper-logged-answer"),
        scorers=[
            cc.RemoteFunction.from_name("juniper-judge"),
            cc.RemoteFunction.from_name("juniper-policy-compliance"),
        ],
        metadata={"window": window, "thresholds": THRESHOLDS},
        suite_id=suite.id,
    )
    run = evaluation.run(metadata={"window": window, "rows": len(turns)})
    report(
        status="running", evaluation_id=evaluation.id, eval_run_id=run.id, window=window
    )

    # A run advances when it is read, and the finalizing read can outlast one RPC deadline;
    # retry the read rather than treating a timeout as a failed run.
    deadline = time.monotonic() + 15 * 60
    while True:
        try:
            run = cc.EvaluationRun.from_id(run.id)
            if run.is_terminal:
                break
        except cc.EvaluationError:
            pass
        if time.monotonic() > deadline:
            report(status="timed_out", finished_at=time.time())
            return
        time.sleep(10)
    status = str(run.status).split(".")[-1].lower()
    if run.result_dataset is None:
        report(status=status, finished_at=time.time())
        return

    table = cc.DatasetClient().read(run.result_dataset)

    def meta(value: object) -> dict:
        if isinstance(value, str):
            try:
                return json.loads(value)
            except ValueError:
                return {}
        return value if isinstance(value, dict) else {}

    rows = []
    for raw in table.to_pylist():
        rows.append(
            {
                "turn_id": raw["turn_id"],
                "question": raw["question"],
                "answer": raw["answer"],
                "prompt_version": raw["prompt_version"],
                "source": raw["source"],
                "judge": raw.get("juniper-judge_value"),
                "judge_reason": meta(raw.get("juniper-judge_metadata")).get(
                    "reason", ""
                ),
                "policy": raw.get("juniper-policy-compliance_value"),
                "violations": meta(raw.get("juniper-policy-compliance_metadata")).get(
                    "violations", []
                ),
            }
        )
    judged = [r["judge"] for r in rows if r["judge"] is not None]
    policed = [r["policy"] for r in rows if r["policy"] is not None]
    metrics = {
        "judge_score_avg": round(sum(judged) / len(judged), 3) if judged else None,
        "policy_pass_rate": round(sum(1 for p in policed if p >= 1.0) / len(policed), 3)
        if policed
        else None,
    }
    versions = sorted({r["prompt_version"] for r in rows})

    breached = [
        m
        for m, limit in THRESHOLDS.items()
        if metrics[m] is not None and metrics[m] < limit
    ]
    events = [
        ("eval.threshold_crossed", m) for m in breached if m not in previously_breached
    ]
    events += [
        ("eval.threshold_recovered", m)
        for m in previously_breached
        if m not in breached
    ]
    deliveries = []
    for event, metric in events:
        worst = sorted(
            rows,
            key=lambda r: (
                (r["policy"] if metric == "policy_pass_rate" else r["judge"]) or 0.0
            ),
        )[:3]
        payload = {
            "event": event,
            "metric": metric,
            "value": metrics[metric],
            "threshold": THRESHOLDS[metric],
            "window": window,
            "rows": len(rows),
            "prompt_versions": versions,
            "evaluation_id": evaluation.id,
            "evaluation_run_id": run.id,
            "examples": [
                {
                    "question": r["question"],
                    "answer": r["answer"][:400],
                    "judge": r["judge"],
                    "judge_reason": r["judge_reason"],
                    "violations": r["violations"],
                }
                for r in worst
            ]
            if event == "eval.threshold_crossed"
            else [],
        }
        body = json.dumps(payload).encode()
        signature = hmac.new(token.encode(), body, hashlib.sha256).hexdigest()
        resp = httpx.post(
            f"{demo}/hooks/eval-alerts",
            content=body,
            headers={
                "content-type": "application/json",
                "x-signature-sha256": signature,
                "x-event": event,
            },
            timeout=30.0,
        )
        deliveries.append(
            {"event": event, "metric": metric, "status": resp.status_code}
        )

    report(
        status=status,
        finished_at=time.time(),
        metrics=metrics,
        breached=breached,
        deliveries=deliveries,
        prompt_versions=versions,
        results=[{**r, "answer": r["answer"][:600]} for r in rows],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default=None)
    parser.add_argument("--demo-url", required=True)
    parser.add_argument(
        "--run-now", action="store_true", help="invoke one window immediately"
    )
    args = parser.parse_args()
    for function in (logged_answer, policy_compliance, judge, auto_eval):
        function.wait_ready(timeout=900)
        print("ready:", function, flush=True)
    if args.run_now:
        auto_eval.remote()
    return 0


if __name__ == "__main__":
    sys.exit(main())
