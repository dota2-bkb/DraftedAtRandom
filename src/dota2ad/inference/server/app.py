"""Litestar app factory + uvicorn entrypoint."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import torch
import uvicorn
from litestar import Litestar, Request, Response
from litestar.exceptions import HTTPException, ValidationException

from dota2ad.core import default_paths, load_vocabs
from dota2ad.models import load_policy
from dota2ad.suggest import load_stats_dqn
from dota2ad.training.weights import DEFAULT_BALANCED_WEIGHTS, get_preset

from dota2ad.inference.server.context import AppContext
from dota2ad.inference.server.lookups import build_lookups
from dota2ad.inference.server.routes import DraftController, LookupsController
from dota2ad.inference.server.schemas import ErrorResponse
from dota2ad.inference.server.state import make_empty_state


PORT = 5000


def bootstrap_ctx() -> AppContext:
    paths = default_paths()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading vocabs...")
    vocabs = load_vocabs()

    print("Loading policy model...")
    policy = load_policy(paths.policy_ckpt, vocabs, device)
    mmr_mean = policy.mmr_mean.item()
    mmr_std = policy.mmr_std.item()

    print("Building lookups...")
    lookups = build_lookups(paths, vocabs, mmr_mean, mmr_std)
    print(
        f"  {len(lookups.hero_id_to_name)} heroes, {len(lookups.ability_id_to_name)} abilities, "
        f"{len(lookups.hero_abilities)} hero->ability sets"
    )

    recommenders: dict = {}
    preset_name = "balanced"
    # Q ("stats-DQN"), overridable via DOTA2AD_STATS_DQN_CKPT without clobbering the
    # canonical models/stats_dqn.pt.
    stats_dqn_ckpt = Path(os.environ.get("DOTA2AD_STATS_DQN_CKPT", str(paths.stats_dqn_ckpt)))
    if stats_dqn_ckpt.exists():
        print(f"Loading Q from {stats_dqn_ckpt}...")
        recommenders["q"] = load_stats_dqn(stats_dqn_ckpt, vocabs, device)
    # Trial — a QNetStats fit on only the randomized picks (unconfounded causal value;
    # experiments/stats-recommender-value/trial.py --save).
    trial_ckpt = paths.models / "trial.pt"
    if trial_ckpt.exists():
        print(f"Loading Trial from {trial_ckpt}...")
        recommenders["trial"] = load_stats_dqn(trial_ckpt, vocabs, device)

    active = "q" if "q" in recommenders else next(iter(recommenders), "bc")
    stat_weights = get_preset(preset_name) if recommenders else DEFAULT_BALANCED_WEIGHTS.clone()
    if recommenders:
        print(f"  recommenders={['bc', *recommenders]}, active='{active}', preset='{preset_name}'")
    else:
        print("No stats-DQN checkpoints found; serving BC softmax only.")

    return AppContext(
        state=make_empty_state(),
        lookups=lookups,
        policy=policy,
        device=device,
        lock=asyncio.Lock(),
        recommenders=recommenders,
        active_recommender=active,
        preset_name=preset_name,
        stat_weights=stat_weights,
    )


async def on_startup(app: Litestar) -> None:
    app.state.ctx = bootstrap_ctx()


def http_exception_handler(_: Request, exc: HTTPException) -> Response[ErrorResponse]:
    return Response(
        content=ErrorResponse(error=exc.detail),
        status_code=exc.status_code,
        media_type="application/json",
    )


def validation_exception_handler(_: Request, exc: ValidationException) -> Response[ErrorResponse]:
    return Response(
        content=ErrorResponse(error=exc.detail),
        status_code=400,
        media_type="application/json",
    )


app = Litestar(
    route_handlers=[DraftController, LookupsController],
    on_startup=[on_startup],
    exception_handlers={
        HTTPException: http_exception_handler,
        ValidationException: validation_exception_handler,
    },
    debug=True,
)


def main() -> None:
    uvicorn.run(
        "dota2ad.inference.server:app",
        host="0.0.0.0",
        port=PORT,
        reload=True,
    )
