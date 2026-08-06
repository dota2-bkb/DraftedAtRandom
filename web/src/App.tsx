import { useDraft } from "@/hooks/useDraft";
import { Controls } from "@/components/Controls";
import { LoadoutColumn } from "@/components/LoadoutColumn";
import { Pool } from "@/components/Pool";
import { Suggestions } from "@/components/Suggestions";

export default function App() {
  const { data, isLoading, error } = useDraft();

  if (isLoading) return <div className="p-4">Loading...</div>;
  if (error) return <div className="p-4 text-accent-error">Error: {error.message}</div>;
  if (!data) return null;

  const { state } = data;
  const radiantSlots = [0, 2, 4, 6, 8];
  const direSlots = [1, 3, 5, 7, 9];

  return (
    <div className="flex h-full flex-col px-2.5 py-1.5 gap-1.5">
      <Controls state={state} />
      <div className="flex flex-1 min-h-0 gap-2">
        <LoadoutColumn
          side="radiant"
          slots={radiantSlots}
          picks={state.player_picks}
          activeSlot={state.pick_slot}
          turn={state.turn}
        />
        <div className="flex flex-1 min-w-0 min-h-0 flex-col gap-1.5">
          <Pool state={state} suggestions={data.suggestions} />
          <Suggestions
            suggestions={data.suggestions}
            confidence={data.confidence}
            preferences={data.preferences}
            recommenders={data.recommenders}
            recommender={data.recommender}
            reweightBeta={data.reweight_beta}
          />
        </div>
        <LoadoutColumn
          side="dire"
          slots={direSlots}
          picks={state.player_picks}
          activeSlot={state.pick_slot}
          turn={state.turn}
        />
      </div>
    </div>
  );
}
