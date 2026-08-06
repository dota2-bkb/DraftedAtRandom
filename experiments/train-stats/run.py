"""Train match-stats model and save to models/match_stats.pt."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from dota2ad.core import (
    StatsRecord,
    build_stats_records,
    compute_stats_norm,
    default_paths,
    load_split,
    load_stats_rows,
    load_vocabs,
    stats_collate,
)
from dota2ad.models import StatsModel

D = 64
BATCH = 1024  # large batch: small model is GPU-launch/overhead-bound at small batch
LR = 1e-3
WD = 1e-2
EPOCHS = 100
PATIENCE = 15

LOSS_KEYS = ["scalar", "gold_t", "xp_t", "lh_t", "matchup", "priority", "damage", "gold_reasons", "xp_reasons", "spell_damage"]


class StatsDataset(Dataset):
    def __init__(self, records: list[StatsRecord]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        return self.records[i]


def compute_losses(
    model: StatsModel,
    batch: dict[str, torch.Tensor],
    mse: nn.MSELoss,
) -> dict[str, torch.Tensor]:
    """Run model forward and compute per-task losses."""
    scalars, gold_t, xp_t, lh_t, matchups, priorities, damage, gold_reasons, xp_reasons, spell_damage = model(batch)

    loss_scalar = mse(scalars, batch["scalar_stats"])

    tmask = batch["time_mask"].float()
    n_valid_3 = (tmask.sum() * 3).clamp(min=1)
    loss_gold = ((gold_t - batch["gold_t"]) ** 2 * tmask).sum() / n_valid_3
    loss_xp = ((xp_t - batch["xp_t"]) ** 2 * tmask).sum() / n_valid_3
    loss_lh = ((lh_t - batch["lh_t"]) ** 2 * tmask).sum() / n_valid_3

    matchup_targets = torch.stack([batch["kill_counts"], batch["death_counts"]], dim=-1)
    loss_matchup = mse(matchups, matchup_targets)

    loss_priority = mse(priorities, batch["ability_priorities"])

    loss_damage = mse(damage, batch["damage_dealt"])
    loss_gold_reasons = mse(gold_reasons, batch["gold_reasons"])
    loss_xp_reasons = mse(xp_reasons, batch["xp_reasons"])
    loss_spell_damage = mse(spell_damage, batch["spell_damage_dealt"])

    return {
        "scalar": loss_scalar,
        "gold_t": loss_gold,
        "xp_t": loss_xp,
        "lh_t": loss_lh,
        "matchup": loss_matchup,
        "priority": loss_priority,
        "damage": loss_damage,
        "gold_reasons": loss_gold_reasons,
        "xp_reasons": loss_xp_reasons,
        "spell_damage": loss_spell_damage,
    }


# ---------------------------------------------------------------------------
# Loss combination strategies
# ---------------------------------------------------------------------------


class SumCombiner:
    """Plain sum of all losses (baseline)."""

    def combine(self, losses: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        return torch.stack(list(losses.values())).sum(), {k: 1.0 for k in losses}

    def extra_param_groups(self) -> list[dict]:
        return []


class KendallCombiner:
    """Learned uncertainty weighting (Kendall et al., 2018)."""

    def __init__(self, device: torch.device):
        self.log_vars = {k: nn.Parameter(torch.zeros(1, device=device)) for k in LOSS_KEYS}

    def combine(self, losses: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        parts = []
        weights: dict[str, float] = {}
        for k in LOSS_KEYS:
            s = self.log_vars[k]
            parts.append((torch.exp(-s) * losses[k] + s) * 0.5)
            weights[k] = torch.exp(-s).item()
        return torch.stack(parts).sum(), weights

    def extra_param_groups(self) -> list[dict]:
        return [{"params": list(self.log_vars.values()), "weight_decay": 0.0}]


class EmaCombiner:
    """Divide each loss by its exponential moving average to normalize scales."""

    def __init__(self, alpha: float = 0.99):
        self.emas: dict[str, float] = {k: 1.0 for k in LOSS_KEYS}
        self.alpha = alpha

    def combine(self, losses: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        scaled = []
        weights: dict[str, float] = {}
        for k in LOSS_KEYS:
            self.emas[k] = self.alpha * self.emas[k] + (1 - self.alpha) * losses[k].item()
            denom = max(self.emas[k], 1e-8)
            scaled.append(losses[k] / denom)
            weights[k] = 1.0 / denom
        return torch.stack(scaled).sum(), weights

    def extra_param_groups(self) -> list[dict]:
        return []


class GradNormCombiner:
    """GradNorm (Chen et al., 2018): balance gradient norms on shared layer."""

    def __init__(self, shared_params, device: torch.device, alpha: float = 0.5):
        self.weights = nn.Parameter(torch.ones(len(LOSS_KEYS), device=device))
        self.shared_params = list(shared_params)
        self.alpha = alpha
        self.initial_losses: dict[str, float] = {}
        self.weight_lr = 0.025

    def combine(self, losses: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        N = len(LOSS_KEYS)
        task_losses = [losses[k] for k in LOSS_KEYS]

        if not self.initial_losses:
            self.initial_losses = {k: max(losses[k].item(), 1e-8) for k in LOSS_KEYS}

        # Per-task gradient norms on shared params
        norms = []
        for i in range(N):
            grads = torch.autograd.grad(
                self.weights[i] * task_losses[i],
                self.shared_params,
                retain_graph=True,
                create_graph=True,
            )
            norm = torch.cat([g.flatten() for g in grads]).norm()
            norms.append(norm)
        norms_t = torch.stack(norms)

        # Targets based on inverse training rate
        mean_norm = norms_t.mean().detach()
        loss_ratios = torch.tensor(
            [losses[k].item() / self.initial_losses[k] for k in LOSS_KEYS],
            device=self.weights.device,
        )
        inv_rates = loss_ratios / loss_ratios.mean()
        targets = mean_norm * inv_rates**self.alpha

        # Update weights via GradNorm loss
        gn_loss = torch.abs(norms_t - targets).sum()
        w_grad = torch.autograd.grad(gn_loss, self.weights, retain_graph=True)[0]
        with torch.no_grad():
            self.weights.data -= self.weight_lr * w_grad
            self.weights.data.clamp_(min=0.1)
            self.weights.data *= N / self.weights.data.sum()

        total = torch.stack([self.weights[i].detach() * task_losses[i] for i in range(N)]).sum()
        eff_weights = {k: self.weights[i].item() for i, k in enumerate(LOSS_KEYS)}
        return total, eff_weights

    def extra_param_groups(self) -> list[dict]:
        return []


class GeoMeanCombiner:
    """Geometric mean of losses — scale-invariant, no learnable params."""

    def combine(self, losses: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        log_losses = torch.stack([torch.log(losses[k].clamp(min=1e-8)) for k in LOSS_KEYS])
        total = torch.exp(log_losses.mean())
        weights = {k: 1.0 / len(LOSS_KEYS) for k in LOSS_KEYS}
        return total, weights

    def extra_param_groups(self) -> list[dict]:
        return []


class RLWCombiner:
    """Random Loss Weighting: sample softmax(randn) weights each batch."""

    def combine(self, losses: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        dev = next(iter(losses.values())).device
        w = torch.softmax(torch.randn(len(LOSS_KEYS), device=dev), dim=0)
        task_losses = torch.stack([losses[k] for k in LOSS_KEYS])
        total = (w * task_losses).sum()
        weights = {k: w[i].item() for i, k in enumerate(LOSS_KEYS)}
        return total, weights

    def extra_param_groups(self) -> list[dict]:
        return []


COMBINERS = {
    "sum": lambda model, device: SumCombiner(),
    "kendall": lambda model, device: KendallCombiner(device),
    "ema": lambda model, device: EmaCombiner(),
    "gradnorm": lambda model, device: GradNormCombiner(model.st_loadout.parameters(), device),
    "geomean": lambda model, device: GeoMeanCombiner(),
    "rlw": lambda model, device: RLWCombiner(),
}


def train(
    *,
    loss_strategy: str = "rlw",
    epochs: int = EPOCHS,
    output_path: Path | None = None,
) -> None:
    """Train match-stats model. Callable from other modules."""
    if output_path is None:
        output_path = default_paths().stats_ckpt
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_match_stats] Device: {device}, loss strategy: {loss_strategy}")

    vocabs = load_vocabs()
    split = load_split()
    all_rows = load_stats_rows()
    print(f"Total rows: {len(all_rows)}")

    train_rows = [r for r in all_rows if r.match_id not in split.held_out]
    val_rows = [r for r in all_rows if r.match_id in split.val_ids]
    print(f"Train: {len(train_rows)}, Val: {len(val_rows)}")

    norm = compute_stats_norm(train_rows)
    train_recs = build_stats_records(train_rows, norm, vocabs)
    val_recs = build_stats_records(val_rows, norm, vocabs)

    train_loader = DataLoader(
        StatsDataset(train_recs), batch_size=BATCH, shuffle=True, collate_fn=stats_collate, pin_memory=True,
    )
    val_loader = DataLoader(
        StatsDataset(val_recs), batch_size=BATCH, shuffle=False, collate_fn=stats_collate, pin_memory=True,
    )

    vocab_size = len(vocabs.draft_id_to_index)
    model = StatsModel(vocab_size, d=D)
    model.set_norm(norm)
    model = model.to(device)

    combiner = COMBINERS[loss_strategy](model, device)

    param_groups = [{"params": model.parameters()}]
    param_groups.extend(combiner.extra_param_groups())
    optimizer = torch.optim.AdamW(param_groups, lr=LR, weight_decay=WD)
    mse = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    # Per-epoch checkpoint/resume (crash-safe). Final model written only on
    # completion, so the run_all.sh stage guard still detects "done". (Default
    # rlw combiner is stateless; non-default combiners' weights reset on resume.)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_path.with_suffix(".ckpt")
    start_epoch = 0
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        best_val_loss = ck["best_val_loss"]
        best_state = ck["best_state"]
        patience_counter = ck["patience_counter"]
        start_epoch = ck["epoch"] + 1
        print(f"Resumed from {ckpt_path}: starting at epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        model.train()
        train_total = 0.0
        train_n = 0
        train_comp = {k: 0.0 for k in LOSS_KEYS}
        train_wt = {k: 0.0 for k in LOSS_KEYS}

        for batch in train_loader:
            b = {k: v.to(device) for k, v in batch.items()}
            losses = compute_losses(model, b, mse)
            B = b["mmr"].shape[0]

            loss, eff_w = combiner.combine(losses)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_total += loss.item() * B
            train_n += B
            for k in LOSS_KEYS:
                train_comp[k] += losses[k].item() * B
                train_wt[k] += eff_w[k] * B

        train_total /= train_n
        for k in LOSS_KEYS:
            train_comp[k] /= train_n
            train_wt[k] /= train_n

        # Validation (always plain sum for comparability)
        model.eval()
        val_total = 0.0
        val_n = 0
        val_comp = {k: 0.0 for k in LOSS_KEYS}
        with torch.no_grad():
            for batch in val_loader:
                b = {k: v.to(device) for k, v in batch.items()}
                losses = compute_losses(model, b, mse)
                B = b["mmr"].shape[0]
                val_total += sum(l.item() for l in losses.values()) * B
                val_n += B
                for k in LOSS_KEYS:
                    val_comp[k] += losses[k].item() * B

        val_total /= val_n
        for k in LOSS_KEYS:
            val_comp[k] /= val_n

        tc = " ".join(f"{k}={train_comp[k]:.4f}" for k in LOSS_KEYS)
        vc = " ".join(f"{k}={val_comp[k]:.4f}" for k in LOSS_KEYS)
        wt = " ".join(f"{k}={train_wt[k]:.3f}" for k in LOSS_KEYS)
        print(f"Epoch {epoch+1:3d}  train={train_total:.4f}  val={val_total:.4f}")
        print(f"  train: {tc}")
        print(f"  val:   {vc}")
        print(f"  weights: {wt}")

        if val_total < best_val_loss:
            best_val_loss = val_total
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        tmp = ckpt_path.with_name(ckpt_path.name + ".tmp")
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "best_state": best_state,
            "patience_counter": patience_counter,
        }, tmp)
        tmp.replace(ckpt_path)

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best_state,
        "d": model.d,
        "n_heads": model.n_heads,
    }, output_path)
    print(f"Saved to {output_path} (best val_loss={best_val_loss:.4f})")
    ckpt_path.unlink(missing_ok=True)  # training complete — drop the resume checkpoint


def main():
    parser = argparse.ArgumentParser(description="Train match-stats model")
    parser.add_argument("--loss", choices=list(COMBINERS), default="rlw")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    train(loss_strategy=args.loss, epochs=args.epochs, output_path=args.output)


if __name__ == "__main__":
    main()
