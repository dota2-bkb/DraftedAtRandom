"""Train behavior policy and save to models/policy.pt."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dota2ad.core import (
    PolicyDataset,
    compute_mmr_norm,
    default_paths,
    labeled_policy_collate,
    load_matches,
    load_split,
    load_vocabs,
)
from dota2ad.models import BehaviorPolicy
from dota2ad.suggest.density import compute_support_quantiles
from dota2ad.eval.results import write_results

D = 64
BATCH = 1024  # large batch: tiny D=64 model is GPU-launch/overhead-bound at small batch
LR = 1e-3
WD = 1e-2
EPOCHS = 100
PATIENCE = 10
CALIB_MATCHES = 400  # held-out matches replayed to calibrate the state-rarity band


def evaluate(model: BehaviorPolicy, loader: DataLoader, device: torch.device, random_class_idx: int):
    model.eval()
    total_loss = 0.0
    correct1 = 0
    correct5 = 0
    total = 0
    # Per-class slice diagnostics (random vs deliberate). The BC always emits
    # the V+1 random/timeout class.
    n_random = 0
    n_random_correct = 0
    random_mean_prob = 0.0
    n_deliberate = 0
    n_deliberate_correct1 = 0
    all_top1_probs: list[torch.Tensor] = []
    all_top1_hits: list[torch.Tensor] = []
    brier_sum = 0.0
    with torch.no_grad():
        for batch in loader:
            b = {k: v.to(device) for k, v in batch.items()}
            log_probs = model(b)
            targets = b["action_idx"]
            loss = nn.functional.nll_loss(log_probs, targets)
            n = targets.shape[0]
            total_loss += loss.item() * n
            # Top-1 / Top-5
            topk = log_probs.topk(5, dim=-1).indices
            hit1 = (topk[:, 0] == targets)
            correct1 += hit1.sum().item()
            correct5 += (topk == targets.unsqueeze(1)).any(dim=1).sum().item()
            total += n
            # Top-label ECE: confidence & correctness of the argmax prediction
            probs = log_probs.exp()
            top1_probs = probs[torch.arange(n, device=device), topk[:, 0]]
            all_top1_probs.append(top1_probs.cpu())
            all_top1_hits.append(hit1.cpu())
            # Brier score: sum of (p - one_hot)^2 over classes
            one_hot = torch.zeros_like(probs).scatter_(1, targets.unsqueeze(1), 1.0)
            brier_sum += ((probs - one_hot) ** 2).sum(dim=1).mean().item() * n
            rand_mask = targets == random_class_idx
            if rand_mask.any():
                n_random += int(rand_mask.sum().item())
                n_random_correct += int(((topk[:, 0] == targets) & rand_mask).sum().item())
                random_mean_prob += float(probs[rand_mask, random_class_idx].sum().item())
            det_mask = ~rand_mask
            if det_mask.any():
                n_deliberate += int(det_mask.sum().item())
                n_deliberate_correct1 += int(((topk[:, 0] == targets) & det_mask).sum().item())

    # Top-label ECE (equal-width bins)
    all_confs = torch.cat(all_top1_probs)
    all_hits = torch.cat(all_top1_hits).float()
    n_bins = 15
    ece = 0.0
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        mask = (all_confs > lo) & (all_confs <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = all_confs[mask].mean().item()
        bin_acc = all_hits[mask].mean().item()
        ece += mask.sum().item() / total * abs(bin_acc - bin_conf)

    slice_stats: dict[str, float] = {}
    if n_random > 0:
        slice_stats["rand_recall"] = n_random_correct / n_random
        slice_stats["rand_mean_prob"] = random_mean_prob / n_random
    if n_deliberate > 0:
        slice_stats["det_top1"] = n_deliberate_correct1 / n_deliberate
    slice_stats["n_random"] = float(n_random)
    slice_stats["n_deliberate"] = float(n_deliberate)
    return total_loss / total, correct1 / total, correct5 / total, ece, brier_sum / total, slice_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--init-from", type=str, default=None,
                        help="warm-start weights from a policy.pt (keeps THIS dataset's mmr norm)")
    parser.add_argument("--seed", type=int, default=None,
                        help="seed init + data order (for a seed sweep); None = unseeded default")
    parser.add_argument("--output", type=str, default=None,
                        help="write the trained policy here instead of models/policy.pt "
                             "(for seed sweeps; the resumable checkpoint sits next to it)")
    args = parser.parse_args()

    if args.seed is not None:
        import random as _random
        torch.manual_seed(args.seed); np.random.seed(args.seed); _random.seed(args.seed)

    paths = default_paths()
    out_ckpt = Path(args.output) if args.output else paths.policy_ckpt
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  seed={args.seed}  out={out_ckpt}")

    matches = load_matches(paths.matches)
    vocabs = load_vocabs(paths.vocabs)
    vocab_size = len(vocabs.draft_id_to_index)
    print(f"Loaded {len(matches)} matches, vocab_size={vocab_size}")

    split = load_split(paths.split)
    train_matches = [m for m in matches if m.match_id not in split.held_out]
    val_matches = [m for m in matches if m.match_id in split.val_ids]
    mmr_mean, mmr_std = compute_mmr_norm(train_matches)
    print(f"Train: {len(train_matches)} matches, Val: {len(val_matches)} matches")

    random_class_idx = vocab_size

    t0 = time.monotonic()
    train_ds = PolicyDataset(
        train_matches, vocabs, mmr_mean, mmr_std, random_class_idx=random_class_idx,
    )
    val_ds = PolicyDataset(
        val_matches, vocabs, mmr_mean, mmr_std, random_class_idx=random_class_idx,
    )
    print(f"Encoded datasets in {time.monotonic() - t0:.1f}s ({len(train_ds)} train, {len(val_ds)} val samples)")
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, collate_fn=labeled_policy_collate, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, collate_fn=labeled_policy_collate, pin_memory=True)

    model = BehaviorPolicy(vocab_size, D)
    model.mmr_mean.fill_(mmr_mean)
    model.mmr_std.fill_(mmr_std)
    model = model.to(device)
    if args.init_from:
        sd = torch.load(args.init_from, map_location=device)["state_dict"]
        sd.pop("mmr_mean", None)
        sd.pop("mmr_std", None)  # keep THIS dataset's (rating) normalization
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"Warm-started from {args.init_from} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    best_val_loss = float("inf")
    best_state = None
    best_metrics: dict | None = None
    patience_counter = 0

    # Per-epoch checkpoint so a crash (WSL/GPU restart) resumes mid-training rather
    # than losing all completed epochs. The final policy.pt is written only on
    # completion, so the run_all.sh stage guard still detects "done" correctly.
    paths.models.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_ckpt.with_suffix(".ckpt")   # resumable checkpoint next to the output
    start_epoch = 0
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        best_val_loss = ck["best_val_loss"]
        best_state = ck["best_state"]
        best_metrics = ck.get("best_metrics")
        patience_counter = ck["patience_counter"]
        start_epoch = ck["epoch"] + 1
        print(f"Resumed from {ckpt_path}: starting at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        t_epoch = time.monotonic()
        model.train()
        train_loss = 0.0
        train_n = 0
        for batch in train_loader:
            b = {k: v.to(device) for k, v in batch.items()}
            log_probs = model(b)
            loss = nn.functional.nll_loss(log_probs, b["action_idx"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * b["action_idx"].shape[0]
            train_n += b["action_idx"].shape[0]

        train_loss /= train_n
        val_loss, val_top1, val_top5, val_ece, val_brier, slice_stats = evaluate(
            model, val_loader, device, random_class_idx=random_class_idx,
        )
        elapsed = time.monotonic() - t_epoch
        msg = (
            f"Epoch {epoch+1:3d}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"top1={val_top1:.3f}  "
            f"top5={val_top5:.3f}  "
            f"ece={val_ece:.4f}  "
            f"brier={val_brier:.4f}  "
            f"({elapsed:.1f}s)"
        )
        if slice_stats:
            extras = []
            if "det_top1" in slice_stats:
                extras.append(f"det_top1={slice_stats['det_top1']:.3f}")
            if "rand_recall" in slice_stats:
                extras.append(f"rand_recall={slice_stats['rand_recall']:.3f}")
            if "rand_mean_prob" in slice_stats:
                extras.append(f"rand_mean_p={slice_stats['rand_mean_prob']:.3f}")
            msg += "  " + " ".join(extras)
        print(msg)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {"val_loss": float(val_loss), "top1": float(val_top1),
                            "top5": float(val_top5), "ece": float(val_ece),
                            "brier": float(val_brier),
                            **{k: float(v) for k, v in slice_stats.items()}}
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
            "best_metrics": best_metrics,
            "patience_counter": patience_counter,
        }, tmp)
        tmp.replace(ckpt_path)

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Calibrate the state-rarity confidence band on held-out val (BC density) and
    # bake it into the checkpoint, so it always travels with this exact policy.
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    support_q = compute_support_quantiles(
        val_matches[:CALIB_MATCHES], model, vocabs, mmr_mean, mmr_std, device)
    print(f"Density band quantiles (10/30/50/70/90) = {[round(s, 2) for s in support_q]}")

    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best_state,
        "d": D,
        "n_heads": model.n_heads,
        "density_support_q": support_q,
    }, out_ckpt)
    print(f"Saved to {out_ckpt} (best val_loss={best_val_loss:.4f})")
    if best_metrics is not None and args.output is None and args.init_from is None and args.seed is None:
        write_results("train-policy", best_metrics)
    ckpt_path.unlink(missing_ok=True)  # training complete — drop the resume checkpoint


if __name__ == "__main__":
    main()
