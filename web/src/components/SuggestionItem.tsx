import type { PreferencesState, SuggestionEntry } from "@/api/client";
import { abilityIcon, heroIcon } from "@/lib/icons";
import { usePick } from "@/hooks/useDraft";

type SortKey = "policy" | "dqn";

interface SuggestionItemProps {
  rank: number;
  suggestion: SuggestionEntry;
  piPct: number;
  qPct: number | null;
  sortBy: SortKey;
  isTopPick: boolean;
  // π below this → "rare here" (a third of uniform; set by the parent from pool size)
  rareThreshold: number;
  // When stats-DQN is loaded the per-suggestion q is in (weighted z-score)
  // units, NOT a probability. Pass preferences in so we format the q
  // correctly and surface the top contributing stats for the top pick.
  preferences?: PreferencesState | null;
  presetWeights?: number[] | null;
}

// Friendly display name for stats keys
function statDisplayName(name: string): string {
  return name
    .replace(/_/g, " ")
    .replace(/\bvs\b/, "vs")
    .replace(/ \(lower = better\)/, "");
}

export function SuggestionItem({
  rank,
  suggestion,
  piPct,
  qPct,
  sortBy,
  isTopPick,
  rareThreshold,
  preferences,
  presetWeights,
}: SuggestionItemProps) {
  const pick = usePick();
  const isHero = suggestion.type === "hero";
  const icon = isHero ? heroIcon(suggestion.name) : abilityIcon(suggestion.name);
  const probPct = (suggestion.prob * 100).toFixed(1);
  // Stats-DQN composite is a weighted z-score sum, not a probability.
  const statsDqnActive = preferences != null && suggestion.q_vec != null;
  const qText = suggestion.q == null
    ? "—"
    : `${suggestion.q >= 0 ? "+" : ""}${suggestion.q.toFixed(2)}z`;
  const sortIsDqn = sortBy === "dqn" && qPct != null;

  // Top *positive* contributors to the composite under the active preset.
  // Each entry is unambiguously a reason in favor of the pick. We display
  // the predicted-stat direction ("high X" / "low X") so the user knows
  // what the model thinks is good about this pick — the preset's weight
  // already encodes whether more or less is better for that stat.
  let topContributors: { name: string; magnitude: number; predictedHigh: boolean }[] = [];
  if (isTopPick && statsDqnActive && suggestion.q_vec && presetWeights && preferences?.stat_names) {
    const contribs = suggestion.q_vec.map((q, k) => {
      const w = presetWeights[k] ?? 0;
      const wq = w * q;
      return {
        name: preferences.stat_names![k] ?? `stat_${k}`,
        magnitude: wq,                // signed: positive = good for this preset
        predictedHigh: q > 0,          // model thinks this pick is above-average for this stat
      };
    }).filter((c) => c.magnitude > 0); // keep only reasons in favor
    contribs.sort((a, b) => b.magnitude - a.magnitude);
    topContributors = contribs.slice(0, 3);
  }
  // "Rarely picked here": low BC prob = humans seldom take this action at THIS
  // point in the draft — (s,a) support, stated as a human-convention fact, NOT
  // a model-reliability claim. Shown under Q-sort on EVERY pick below the
  // (pool-relative) threshold, not just the top, so it's consistent: same π →
  // same label. The opening pool is wide, so more picks qualify early — that's
  // honest (most specific opening picks are unconventional).
  const rareHere = sortIsDqn && suggestion.prob < rareThreshold;

  return (
    <div
      className="flex flex-col cursor-pointer rounded bg-bg-cell px-1.5 py-0.5 mb-0.5 text-xs hover:bg-bg-cell-hover"
      onClick={() => pick.mutate({ id: suggestion.id, type: suggestion.type, is_random: false })}
    >
      <div className="flex items-center gap-1">
      <span className="min-w-4 text-text-dim text-[11px]">{rank}.</span>
      <img
        src={icon}
        alt={suggestion.name}
        className={`rounded-sm ${isHero ? "h-5 w-[34px] object-cover" : "h-5 w-5"}`}
        onError={(e) => (e.currentTarget.style.display = "none")}
      />
      <span className="flex-1 truncate">{suggestion.name}</span>
      {rareHere && (
        <span
          className="rounded border border-text-dim px-1 py-0.5 text-[9px] font-semibold text-text-dim"
          title={`Rarely picked here — humans take this at this point in the draft only ${probPct}% of the time. The model ranks it highly anyway, so it diverges from how people usually draft; worth a closer look.`}
        >
          rare here
        </span>
      )}
      <span className={`text-[8px] ${sortIsDqn ? "text-text-dim" : "text-accent"}`}>π</span>
      {!sortIsDqn && (
        <div className="h-2 w-10 overflow-hidden rounded bg-text-very-dim">
          <div className="h-full bg-accent" style={{ width: `${piPct}%` }} />
        </div>
      )}
      <span className={`min-w-[38px] text-right text-xs ${sortIsDqn ? "text-text-muted" : "text-text-fg"}`}>
        {probPct}%
      </span>
      <span className={`text-[8px] ${sortIsDqn ? "text-accent" : "text-text-dim"}`}>Q</span>
      {sortIsDqn && (
        <div className="h-2 w-10 overflow-hidden rounded bg-text-very-dim">
          <div className="h-full bg-accent" style={{ width: `${qPct}%` }} />
        </div>
      )}
      <span className={`min-w-[42px] text-right text-xs ${sortIsDqn ? "text-text-fg" : "text-text-muted"}`}>
        {qText}
      </span>
      </div>
      {topContributors.length > 0 && (
        <div
          className="ml-5 mt-0.5 text-[9px] text-text-dim"
          title={
            "Top three stats contributing positively to this pick's composite score " +
            "under the active preset. ↑ = model predicts above-average for this " +
            "pick; ↓ = below-average. The list is already filtered to reasons IN " +
            "FAVOR of the pick — whether high or low is desirable for each stat is " +
            "encoded in the preset's weights, so every arrow here is a 'good' arrow."
          }
        >
          <span className="opacity-70">Why:</span>{" "}
          {topContributors.map((c, i) => (
            <span key={c.name}>
              {i > 0 && ", "}
              <span className="text-accent">
                {c.predictedHigh ? "↑" : "↓"}
                {statDisplayName(c.name).replace(/\/min$/, "")}
              </span>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
