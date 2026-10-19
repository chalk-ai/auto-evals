# Automatic evaluations with an LLM judge, a scoring function and webhook alerts

A support assistant answers live traffic. Every five minutes a scheduled
[`chalkcompute`](https://pypi.org/project/chalkcompute/) function scores the latest window of
that traffic as a Chalk evaluation. It uses two scorers, an LLM-as-judge and a custom scoring
function, and posts a signed webhook when a window's score crosses a threshold. Ship the v2
prompt, the same prompt built from last season's handbook, and watch the next window catch it.

The assistant, Juniper, works for **Larkspur Outfitters**, a fictional outdoor-gear shop, and
answers from a short policy handbook.

```
shoppers ──▶ Juniper (scaling group) ──▶ traffic log
                                             │  every 5 min
                                             ▼
                      juniper-auto-eval (scheduled function)
                        upload window as a Chalk dataset
                        Chalk evaluation: task = the answer already sent
                          ├─ juniper-judge              LLM-as-judge, via the model gateway
                          └─ juniper-policy-compliance  hard rules, in Python
                        threshold crossed? ──▶ signed webhook ──▶ /hooks/eval-alerts
```

## Files

| File | Role |
| --- | --- |
| `agent.py` | The assistant, the current and last-season handbooks, the v1 and v2 system prompts, and the policy rules the scoring function checks. |
| `traffic.py` | Synthetic shopper questions that keep traffic flowing. |
| `server.py` | FastAPI app: the chat API, the traffic log, internal endpoints for the evaluator, and the webhook receiver. |
| `index.html` | The demo page: live traffic, evaluation runs against their thresholds, and received webhooks. |
| `deploy.py` | Deploys `server.py` as a `chalkcompute.ScalingGroup`. |
| `evaluator.py` | Deploys the evaluation task, both scorers and the scheduled evaluator as Chalk functions. |

## Run it

You need [`uv`](https://docs.astral.sh/uv/) and a Chalk environment with the model gateway
enabled. Create a gateway key with `chalk router api-key create` and save it as the secret
`CHALK_ROUTER_API_KEY`. Credentials come from the SDK’s ambient credentials from `chalk login`, or from `CHALK_API_SERVER`,
`CHALK_CLIENT_ID`, `CHALK_CLIENT_SECRET` and `CHALK_ENVIRONMENT_ID`.

```bash
export AUTO_EVALS_TOKEN=$(openssl rand -hex 24)   # shared by the receiver and the evaluator
./deploy.py                                       # prints the web_url
./evaluator.py --demo-url https://<web_url>
```

The evaluator runs on a `*/5 * * * *` cron. Open the `web_url`, ask Juniper something, and press
**Ship v2 prompt**. The v2 prompt rolls itself back after twelve minutes.

## Scoring

| Scorer | Kind | Score |
| --- | --- | --- |
| `juniper-judge` | `chalkcompute.scorers.llm_judge` | 1.0 consistent with the handbook, 0.5 vague or with a minor unsupported detail, 0.0 contradicts it (an invented refund, code or exception, a wrong number, or a request for payment details). |
| `juniper-policy-compliance` | custom `@chalkcompute.function` scorer | 1.0 unless the answer offers a discount code other than `TRAIL10`, asks for card details in chat, or promises a refund for a purchase past the 45-day window. |

The task, `juniper-logged-answer`, returns the answer production already sent, so the scorers
grade exactly what the shopper saw rather than a fresh replay. Each window is its own
evaluation in the `juniper-auto-evals` suite.

## Thresholds and webhooks

| Metric | Alert when below |
| --- | --- |
| `judge_score_avg` | 0.75 |
| `policy_pass_rate` | 0.90 |

Alerts fire on transitions: `eval.threshold_crossed` when a metric drops below its threshold,
and `eval.threshold_recovered` when it comes back. The body is JSON with the metric, value,
threshold, window, evaluation and run ids, and up to three of the worst examples. The
`x-signature-sha256` header carries the HMAC-SHA256 of the body, keyed with `AUTO_EVALS_TOKEN`.
`server.py`'s `/hooks/eval-alerts` verifies it. Point the evaluator at any receiver that does
the same.

## Notes

- **Shared demo:** the traffic log, run history and received webhooks live in one replica's
  memory, so everyone using the page sees the same state, and a restart clears it.
- **Public page:** chat and prompt switches are rate limited per visitor, and messages are
  capped at 400 characters.
- **Import-time deploy:** `evaluator.py`'s decorators deploy at import time. Every function pod
  imports the whole module, so module-level imports must exist in every function image.

## License

Apache-2.0.

## Local authentication

The SDK reads credentials, API host, and the selected environment from your existing
`chalk login` in `~/.chalk.yml`, including project-scoped and default logins.
For unattended runs, `CHALK_CLIENT_ID` and `CHALK_CLIENT_SECRET` remain supported. Where available, `--env-id`
overrides the target environment; otherwise the SDK uses your login or environment settings.
