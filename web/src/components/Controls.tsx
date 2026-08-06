import { useState } from "react";
import type { AnnotatedDraft } from "@/api/client";
import { useReset, useUndo } from "@/hooks/useDraft";

interface ControlsProps {
  state: AnnotatedDraft;
}

export function Controls({ state }: ControlsProps) {
  const reset = useReset();
  const undo = useUndo();
  const [error, setError] = useState<string | null>(null);

  const team = state.pick_slot % 2 === 0 ? "Radiant" : "Dire";
  const teamSeat = Math.floor(state.pick_slot / 2) + 1;

  return (
    <div className="flex items-center gap-2 flex-shrink-0">
      <button
        className="px-2 py-0.5 rounded bg-text-very-dim hover:bg-text-dim text-xs"
        onClick={() =>
          reset.mutate(undefined, {
            onError: (e) => setError(e.message),
            onSuccess: () => setError(null),
          })
        }
      >
        Reset
      </button>
      <button
        className="px-2 py-0.5 rounded bg-text-very-dim hover:bg-text-dim text-xs"
        onClick={() =>
          undo.mutate(undefined, {
            onError: (e) => setError(e.message),
            onSuccess: () => setError(null),
          })
        }
      >
        Undo
      </button>
      <span className="text-xs text-text-muted">
        Turn {state.turn}/50 · P{state.pick_slot} ({team} #{teamSeat})
      </span>
      {error && <span className="text-xs text-accent-error">{error}</span>}
    </div>
  );
}
