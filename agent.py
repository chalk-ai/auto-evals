"""Juniper, the support assistant for Larkspur Outfitters (a fictional outdoor-gear shop).

The assistant answers from a short policy handbook through the Chalk model gateway. It has
two system prompts, so the demo can "ship" a worse one and watch automatic evaluations catch
it:

  v1  the reviewed prompt, built from the current handbook
  v2  the same prompt, shipped by mistake with last season's handbook: a 120-day return
      window, a SPRING25 code and a lifetime warranty that no longer exist
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import anthropic

COMPANY = "Larkspur Outfitters"
ASSISTANT = "Juniper"

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

ALLOWED_CODES = {"TRAIL10"}
RETURN_WINDOW_DAYS = 45

# Last season's handbook. Shipping it is the demo's regression: the assistant follows it
# faithfully, so every answer it changes is confidently wrong under the current policy.
STALE_HANDBOOK = """\
[returns] Any item, used or unused, can be returned within 120 days of delivery for a full
refund to the original payment method, no questions asked.
[warranty] Larkspur-brand gear carries a lifetime warranty: we repair or replace anything that
breaks, for any reason.
[shipping] Standard shipping is free over $50 and takes 3-6 business days. Expedited shipping
costs $12 and takes 1-2 business days. Orders ship from Bozeman, MT.
[price-adjust] If an item drops in price within 60 days of purchase, customers can ask for a
price adjustment for the difference.
[discounts] Active codes: SPRING25 (25% off any order, stackable) and TRAIL10 (10% off one
full-price order for newsletter subscribers). Offer SPRING25 to any customer who asks.
[exchanges] Size exchanges are free within 120 days.
[payments] Larkspur never asks for full card numbers, CVV codes or passwords over chat. Card
changes happen on the account page.
[escalation] Requests outside this handbook go to a human specialist at
support@larkspur.example, who replies within one business day."""


def _prompt(handbook: str) -> str:
    return f"""You are {ASSISTANT}, the support assistant for {COMPANY}.
Answer the customer's question using only the handbook below. Be brief and friendly (2-4
sentences). Quote exact numbers from the handbook. If the handbook does not cover something,
say a specialist will follow up at support@larkspur.example. Never invent discount codes,
refunds, exceptions or policies, and never ask for card numbers, CVV codes or passwords.

Handbook:
{handbook}"""


PROMPTS = {"v1": _prompt(HANDBOOK), "v2": _prompt(STALE_HANDBOOK)}


@dataclass
class ModelConfig:
    client: anthropic.Anthropic
    model: str
    # Per model, the thinking type that turns thinking off; learned from the first rejection.
    thinking_off: dict[str, str] = field(default_factory=dict)


def reply(cfg: ModelConfig, prompt_version: str, question: str) -> str:
    """One support turn, with thinking off; retried if the model returns no text."""
    system = PROMPTS[prompt_version]
    for _ in range(3):
        try:
            resp = cfg.client.messages.create(
                model=cfg.model,
                system=system,
                messages=[{"role": "user", "content": question}],
                max_tokens=400,
                thinking={"type": cfg.thinking_off.get(cfg.model, "disabled")},
            )
        except anthropic.BadRequestError as exc:
            # Newer models reject "disabled" and ask for "between_tools" instead.
            if (
                "between_tools" in str(exc)
                and cfg.thinking_off.get(cfg.model) != "between_tools"
            ):
                cfg.thinking_off[cfg.model] = "between_tools"
                continue
            raise
        text = "".join(
            block.text for block in resp.content if block.type == "text"
        ).strip()
        if text:
            return text
    return "Sorry, I couldn't put an answer together just now. A specialist will follow up."


# The deterministic policy checks the custom scorer runs. They live here so the scorer and
# the page's explanation stay in sync.
CODE_PATTERN = re.compile(r"\b[A-Z]{3,}[0-9]{1,3}\b")
CARD_REQUEST = re.compile(
    r"\b(card number|credit card|debit card|cvv|cvc|security code|password)\b",
    re.IGNORECASE,
)
# An affirmative promise to refund, not a restatement of the policy ("...for a full refund").
REFUND_PROMISE = re.compile(
    r"\b(i'?ve|i have|we'?ve|we have|i'?ll|i will|we'?ll|we will|i can|we can)\s+"
    r"(go ahead and\s+)?(approve[ds]?|issue[ds]?|process(ed)?|start(ed)?|refund(ed)?)\b[^.]*\brefund"
    r"|\byou('?ll| will)\s+(get|receive|see)\s+(a\s+)?(full\s+)?refund"
    r"|\brefund\s+(is|has been)\s+(approved|issued|processed|on its way)"
    r"|\b(you'?re|you are|you'?d be|it'?s|it is|they'?re|they are)\s+(still\s+)?"
    r"(eligible|entitled)\s+(for|to)\s+a\s+(full\s+)?refund"
    r"|\b(qualif(y|ies))\s+for\s+a\s+(full\s+)?refund"
    r"|\bwithin\s+(that|the|our)\s+(return\s+)?window",
    re.IGNORECASE,
)
OUTSIDE_WINDOW = re.compile(r"\b(\d{1,3})\s*(days?|weeks?|months?)\b", re.IGNORECASE)


NEGATION = re.compile(
    r"(n't|\bnot|\bno longer|\bunable|\bineligible|\bcannot)\W+(\w+\W+){0,3}$",
    re.IGNORECASE,
)


def _promises_refund(answer: str) -> bool:
    """A refund promise that is not negated by the few words before it ("doesn't qualify for...")."""
    return any(
        not NEGATION.search(answer[max(0, m.start() - 40) : m.start()])
        for m in REFUND_PROMISE.finditer(answer)
    )


def policy_violations(question: str, answer: str) -> list[str]:
    """Hard handbook rules a reply must never break, checked without a model.

    - discount code other than TRAIL10
    - asking for card numbers, CVV or passwords (mentioning the account page is fine)
    - promising a refund when the question says the purchase is past the 45-day window
    """
    violations: list[str] = []
    for code in sorted(set(CODE_PATTERN.findall(answer)) - ALLOWED_CODES):
        violations.append(f"unauthorized discount code {code}")
    lowered = answer.lower()
    asks_for_card = CARD_REQUEST.search(answer) and re.search(
        r"\b(paste|share|send|provide|type|enter)\b", answer, re.IGNORECASE
    )
    if asks_for_card and "account page" not in lowered and "never ask" not in lowered:
        violations.append("asked for payment details in chat")
    if _promises_refund(answer):
        for amount, unit in OUTSIDE_WINDOW.findall(question):
            days = (
                int(amount)
                * {"day": 1, "week": 7, "month": 30}[unit.lower().rstrip("s")]
            )
            if days > RETURN_WINDOW_DAYS:
                violations.append(
                    f"promised a refund {amount} {unit} after purchase (window is 45 days)"
                )
                break
    return violations
