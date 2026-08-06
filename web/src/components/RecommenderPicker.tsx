import { useEffect, useState } from "react";
import { useSetPreset } from "@/hooks/useDraft";

interface RecommenderPickerProps {
  recommenders: string[];
  active: string;
  reweightBeta?: number;
}

const LABELS: Record<string, string> = {
  bc: "BC",
  q: "Q",
  trial: "Trial",
  reweight_bc: "BC+Trial",
};

const TOOLTIPS: Record<string, string> = {
  bc:
    "BC — behavior clone: the human-consensus pick policy (predicts what players actually " +
    "pick). Ranks suggestions by BC softmax.",
  q:
    "Q — the ability's value estimated from ALL picks, identified by modeling (a " +
    "continuation / g-formula estimate: causal in intent, but only partially identified). " +
    "Ranks by predicted composite stat value.",
  trial:
    "Trial — the SAME value, but identified from the game's randomized timeout picks (a " +
    "natural experiment — unconfounded). Ranks by predicted composite stat value.",
  reweight_bc:
    "BC+Trial — the human-consensus BC policy tilted toward the Trial value: " +
    "π ∝ π_BC · exp(β · value). β = 0 is pure BC (safe floor); higher β leans harder on the " +
    "randomized-picks evidence. A moderate tilt matches Q.",
};

// Picks WHICH model computes each action's composite Q (and, when it's not BC, the default
// ranking). For reweight-BC, a β slider tunes how hard BC is tilted toward the causal value.
export function RecommenderPicker({ recommenders, active, reweightBeta }: RecommenderPickerProps) {
  const setPreset = useSetPreset();
  const [beta, setBeta] = useState(reweightBeta ?? 1);
  // Keep the slider in sync when the server-confirmed β changes (e.g. after a commit).
  useEffect(() => {
    if (reweightBeta != null) setBeta(reweightBeta);
  }, [reweightBeta]);

  if (recommenders.length <= 1) return null;
  const commit = () => setPreset.mutate({ reweight_beta: beta });

  return (
    <div className="mb-1 flex flex-col gap-1 text-[10px]">
      <div className="flex flex-wrap items-center gap-1">
        <span className="font-semibold text-accent">Model:</span>
        {recommenders.map((r) => {
          const isActive = r === active;
          return (
            <button
              type="button"
              key={r}
              onClick={() => setPreset.mutate({ recommender: r })}
              disabled={setPreset.isPending}
              className={`rounded px-1.5 py-0.5 transition ${
                isActive
                  ? "bg-accent text-bg-base"
                  : "bg-bg-cell text-text-dim hover:bg-bg-cell-hover"
              } ${setPreset.isPending ? "opacity-60" : ""}`}
              title={TOOLTIPS[r] ?? r}
            >
              {LABELS[r] ?? r}
            </button>
          );
        })}
      </div>
      {active === "reweight_bc" && (
        <div className="flex items-center gap-1.5" title={TOOLTIPS.reweight_bc}>
          <span className="text-text-dim">tilt&nbsp;β</span>
          <span className="text-text-very-dim">BC</span>
          <input
            type="range"
            min={0}
            max={8}
            step={0.25}
            value={beta}
            onChange={(e) => setBeta(Number(e.target.value))}
            onPointerUp={commit}
            onKeyUp={commit}
            disabled={setPreset.isPending}
            className="h-1 flex-1 accent-accent"
          />
          <span className="text-text-very-dim">causal</span>
          <span className="w-8 tabular-nums text-text-fg">{beta.toFixed(2)}</span>
        </div>
      )}
    </div>
  );
}
