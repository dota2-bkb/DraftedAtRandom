import type { PreferencesState } from "@/api/client";
import { useSetPreset } from "@/hooks/useDraft";

interface PresetPickerProps {
  preferences: PreferencesState;
}

// Presets are preference knobs (which stats to weight), not win-optimizers.
// Recommender quality is compared on the model picker via β̂-vs-BC.
const PRESET_TOOLTIPS: Record<string, string> = {
  balanced:
    "Default hand-picked ±1 composite over the stats — a broad, all-around weighting.",
  kill_focused:
    "Emphasizes kills, hero damage, stuns, teamfight participation; penalizes deaths. " +
    "Biases suggestions toward high-impact teamfight loadouts.",
  farm_focused:
    "Emphasizes GPM, XPM, last-hits, denies. Biases toward greedy farm patterns.",
  support:
    "Emphasizes assists, healing, stuns, teamfight participation. Biases toward enabling " +
    "teammates over personal output.",
  push:
    "Emphasizes tower damage and tower kills with some GPM. Biases toward objective pressure.",
};

export function PresetPicker({ preferences }: PresetPickerProps) {
  const setPreset = useSetPreset();
  return (
    <div className="mb-1 flex flex-wrap items-center gap-1 text-[10px]">
      <span className="font-semibold text-accent">Preset:</span>
      {preferences.presets.map((p) => {
        const active = p.name === preferences.preset;
        return (
          <button
            type="button"
            key={p.name}
            onClick={() => setPreset.mutate({ preset: p.name })}
            disabled={setPreset.isPending}
            className={`rounded px-1.5 py-0.5 transition ${
              active
                ? "bg-accent text-bg-base"
                : "bg-bg-cell text-text-dim hover:bg-bg-cell-hover"
            } ${setPreset.isPending ? "opacity-60" : ""}`}
            title={PRESET_TOOLTIPS[p.name] ?? p.name}
          >
            {p.name.replace(/_/g, " ")}
          </button>
        );
      })}
    </div>
  );
}
