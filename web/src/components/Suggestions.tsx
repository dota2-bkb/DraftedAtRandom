import { useMemo, useState } from "react";
import type { ConfidenceInfo, PreferencesState, SuggestionEntry } from "@/api/client";
import { SuggestionItem } from "./SuggestionItem";
import { PresetPicker } from "./PresetPicker";
import { RecommenderPicker } from "./RecommenderPicker";

type SortKey = "policy" | "dqn";

interface SuggestionsProps {
  suggestions: SuggestionEntry[];
  confidence: ConfidenceInfo;
  preferences?: PreferencesState | null;
  recommenders?: string[];
  recommender?: string;
  reweightBeta?: number;
}

// Confidence in the recommendations, from the joint draft density: how typical
// the whole draft is (-log p(s)/T under the behavior policy). Validated — unusual
// drafts are where the StatsModel extrapolates, so the recs are less reliable.
const CONF_BAND_LABEL: Record<NonNullable<ConfidenceInfo["state_band"]>, string> = {
  high: "Typical draft",
  medium: "Somewhat unusual",
  low: "Unusual draft",
};

const CONF_BAND_CLASS: Record<NonNullable<ConfidenceInfo["state_band"]>, string> = {
  high: "bg-accent text-bg-base",
  medium: "bg-bg-cell-hover text-text-fg",
  low: "bg-bg-cell text-text-dim",
};

const CONF_BAND_TOOLTIP: Record<NonNullable<ConfidenceInfo["state_band"]>, string> = {
  high:
    "This draft is typical of the training data — the model has seen many like it, " +
    "so the recommendations are on solid ground.",
  medium:
    "This draft is somewhat unusual — the model is starting to extrapolate, so the " +
    "recommendations are moderately reliable.",
  low:
    "This draft is unusual (rare under human play) — the model is extrapolating, " +
    "so weigh the recommendations more cautiously.",
};

export function Suggestions({
  suggestions,
  confidence,
  preferences,
  recommenders,
  recommender,
  reweightBeta,
}: SuggestionsProps) {
  const [filter, setFilter] = useState("");
  const dqnAvailable = useMemo(
    () => suggestions.some((s) => s.q != null),
    [suggestions],
  );
  // Default to Q ranking when available — the recommender's value ordering;
  // π is the human-consensus ordering.
  const [sortBy, setSortBy] = useState<SortKey>(dqnAvailable ? "dqn" : "policy");
  const effectiveSort: SortKey = dqnAvailable ? sortBy : "policy";

  const sorted = useMemo(() => {
    const xs = [...suggestions];
    if (effectiveSort === "policy") {
      xs.sort((a, b) => b.prob - a.prob);
    } else {
      // "dqn" ranks by composite q descending
      xs.sort((a, b) => (b.q ?? -Infinity) - (a.q ?? -Infinity));
    }
    return xs;
  }, [suggestions, effectiveSort]);

  const filtered = useMemo(() => {
    const q = filter.toLowerCase();
    if (!q) return sorted;
    return sorted.filter((sg) => sg.name.toLowerCase().includes(q));
  }, [sorted, filter]);

  const maxProb = filtered.reduce((m, sg) => Math.max(m, sg.prob), 0) || 1;
  const qVals = filtered.map((sg) => sg.q).filter((v): v is number => v != null);
  const qMin = qVals.length ? Math.min(...qVals) : 0;
  const qMax = qVals.length ? Math.max(...qVals) : 1;
  const qSpan = qMax - qMin || 1;
  // "rare here" bar: π below a third of uniform (1/nFeasible). Relative to the
  // pool size so it stays meaningful from the wide opening to the late draft,
  // and applied to EVERY pick (not just the top) so it's consistent.
  const rareThreshold = 1 / (3 * Math.max(1, suggestions.length));

  return (
    <div className="flex flex-1 flex-col overflow-hidden rounded-md bg-bg-card p-1.5 min-h-0">
      <div className="mb-1 flex items-center justify-between gap-2">
        <div className="flex items-center gap-1.5">
          <h2 className="text-[11px] font-semibold text-accent">Suggestions</h2>
          {confidence.state_band && (
            <span
              className={`rounded px-1.5 py-0.5 text-[10px] font-semibold ${CONF_BAND_CLASS[confidence.state_band]}`}
              title={CONF_BAND_TOOLTIP[confidence.state_band]}
            >
              {CONF_BAND_LABEL[confidence.state_band]}
            </span>
          )}
        </div>
        <div className="flex gap-0.5 text-[10px]">
          <button
            type="button"
            onClick={() => setSortBy("policy")}
            className={`rounded px-1.5 py-0.5 ${
              effectiveSort === "policy"
                ? "bg-accent text-bg-base"
                : "bg-bg-cell text-text-dim hover:bg-bg-cell-hover"
            }`}
            title="Sort by BehaviorPolicy probability (predicts human picks)"
          >
            π
          </button>
          <button
            type="button"
            onClick={() => dqnAvailable && setSortBy("dqn")}
            disabled={!dqnAvailable}
            className={`rounded px-1.5 py-0.5 ${
              effectiveSort === "dqn"
                ? "bg-accent text-bg-base"
                : "bg-bg-cell text-text-dim hover:bg-bg-cell-hover"
            } ${!dqnAvailable ? "cursor-not-allowed opacity-40 hover:bg-bg-cell" : ""}`}
            title={
              dqnAvailable
                ? "Sort by MC-Q composite (fast, learned)"
                : "DQN model not loaded"
            }
          >
            Q
          </button>
        </div>
      </div>
      {recommenders && recommender && (
        <RecommenderPicker
          recommenders={recommenders}
          active={recommender}
          reweightBeta={reweightBeta}
        />
      )}
      {preferences && <PresetPicker preferences={preferences} />}
      <input
        value={filter}
        onChange={(e) => setFilter(e.target.value)}
        placeholder="Filter..."
        className="mb-1 w-full flex-shrink-0 rounded border border-text-very-dim bg-bg-cell px-2 py-1 text-xs outline-none"
      />
      <div className="flex-1 overflow-y-auto min-h-0">
        {filtered.length === 0 ? (
          <div className="p-2 text-xs text-text-dim">No suggestions available</div>
        ) : (
          filtered.map((sg, i) => (
            <SuggestionItem
              key={`${sg.type}:${sg.id}`}
              rank={i + 1}
              suggestion={sg}
              piPct={(sg.prob / maxProb) * 100}
              qPct={sg.q != null ? ((sg.q - qMin) / qSpan) * 100 : null}
              sortBy={effectiveSort === "policy" ? "policy" : "dqn"}
              isTopPick={i === 0}
              rareThreshold={rareThreshold}
              preferences={preferences}
              presetWeights={preferences?.active_weights ?? null}
            />
          ))
        )}
      </div>
    </div>
  );
}
