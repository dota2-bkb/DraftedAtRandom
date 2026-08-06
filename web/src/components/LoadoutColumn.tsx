import type { AnnotatedPlayerPick } from "@/api/client";
import { LoadoutCard } from "./LoadoutCard";

interface LoadoutColumnProps {
  side: "radiant" | "dire";
  slots: number[];
  picks: { [key: string]: AnnotatedPlayerPick };
  activeSlot: number;
  turn: number;
}

export function LoadoutColumn({ side, slots, picks, activeSlot, turn }: LoadoutColumnProps) {
  const sideClass =
    side === "radiant"
      ? "border-l-2 border-radiant pl-1"
      : "border-r-2 border-dire pr-1";

  return (
    <div className={`flex w-[170px] flex-shrink-0 flex-col gap-1 ${sideClass}`}>
      {slots.map((slot) => (
        <LoadoutCard
          key={slot}
          slot={slot}
          pick={picks[String(slot)]}
          active={slot === activeSlot && turn < 50}
        />
      ))}
    </div>
  );
}
