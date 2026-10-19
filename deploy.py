#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkcompute==2.13.5"]
# ///
"""Deploy the Juniper support assistant, its traffic log and the webhook receiver as a ScalingGroup.

    export AUTO_EVALS_TOKEN=$(openssl rand -hex 24) # shared with evaluator.py
    ./deploy.py

Requirements on the target environment:
  - the Chalk router is enabled, and a secret holds a router API key (CHALK_ROUTER_API_KEY)
Credentials for the deploy come from your existing `chalk login` (or CHALK_* vars).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP_FILES = ["agent.py", "traffic.py", "server.py", "index.html"]


def main() -> None:
    if not os.environ.get("AUTO_EVALS_TOKEN"):
        raise SystemExit(
            "set AUTO_EVALS_TOKEN (e.g. openssl rand -hex 24); evaluator.py needs the same value"
        )

    from chalkcompute import Image, ReadinessProbe, ScalingGroup, Secret

    image = Image.debian_slim(python_version="3.12").pip_install(
        ["anthropic==1.13.0", "fastapi==0.116.1", "uvicorn[standard]==0.35.0"]
    )
    for name in APP_FILES:
        image = image.add_local_file(
            str(HERE / name), f"/app/{name}", strategy="volume"
        )
    image = image.workdir("/app")

    sg = ScalingGroup(
        image=image,
        name='auto-evals',
        cpu="1",
        memory="2Gi",
        env={
            "AGENT_MODEL": 'anthropic/claude-haiku-5-5',
            "PORT": "8080",
            "ROUTER_API_KEY": Secret.from_chalk_env('CHALK_ROUTER_API_KEY'),
            "AUTO_EVALS_TOKEN": Secret.from_local_env("AUTO_EVALS_TOKEN"),
        },
        entrypoint=["python", "/app/server.py"],
        port=8080,
        readiness_probe=ReadinessProbe.http("/healthz"),
        # The traffic log, run history and received webhooks live in this replica's memory.
        min_replicas=1,
        max_replicas=1,
    )
    t0 = time.monotonic()
    try:
        sg.deploy(build_timeout=1200.0, ready_timeout=900.0)
    finally:
        print(f"deploy returned after {time.monotonic() - t0:.0f}s")
    print("web_url:", sg.web_url)


if __name__ == "__main__":
    main()
