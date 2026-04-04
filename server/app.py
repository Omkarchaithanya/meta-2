"""Manual OpenEnv app entrypoint for the SME negotiation environment."""

from __future__ import annotations

import os

from openenv.core import create_app

from sme_negotiator_env.models import NegotiationAction, NegotiationObservation

from .concurrency import OpenEnvConcurrencyLimiter, max_concurrent_envs_from_env
from .sme_environment import SMENegotiatorEnvironment


_max = max_concurrent_envs_from_env()

app = create_app(
    SMENegotiatorEnvironment,
    NegotiationAction,
    NegotiationObservation,
    env_name="sme-negotiator",
    max_concurrent_envs=_max,
)

app.add_middleware(OpenEnvConcurrencyLimiter, max_concurrent=_max)


def main() -> None:
    """Run the server with uvicorn."""

    import uvicorn

    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "7860")),
    )


if __name__ == "__main__":
    main()
