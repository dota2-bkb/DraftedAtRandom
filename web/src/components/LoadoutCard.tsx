import type { AnnotatedPlayerPick } from "@/api/client";
import { abilityIcon, heroIcon } from "@/lib/icons";

interface LoadoutCardProps {
  slot: number;
  pick: AnnotatedPlayerPick;
  active: boolean;
}

export function LoadoutCard({ slot, pick, active }: LoadoutCardProps) {
  const heroName = pick.hero_name ?? "—";

  return (
    <div
      className={`flex flex-1 flex-col justify-center rounded bg-bg-card p-1.5 text-xs ${
        active ? "outline-2 outline-accent" : ""
      }`}
    >
      <div className="flex items-center gap-1">
        <span className="text-text-very-dim text-[10px]">#{slot}</span>
        {pick.hero_name ? (
          <img
            src={heroIcon(pick.hero_name)}
            className="h-6 w-11 rounded-sm object-cover"
            alt={pick.hero_name}
            onError={(e) => (e.currentTarget.style.visibility = "hidden")}
          />
        ) : (
          <div className="h-6 w-11 rounded-sm bg-bg-cell" />
        )}
        <span className="text-[11px] text-purple-400 truncate">{heroName}</span>
      </div>
      <div className="flex gap-0.5 mt-1">
        {[0, 1, 2].map((i) => {
          const name = pick.basics_names[i];
          return name ? (
            <img
              key={i}
              src={abilityIcon(name)}
              className="h-6 w-6 rounded-sm"
              alt={name}
              title={name}
              onError={(e) => (e.currentTarget.style.visibility = "hidden")}
            />
          ) : (
            <div key={i} className="h-6 w-6 rounded-sm bg-bg-cell" />
          );
        })}
        {pick.ult_name ? (
          <img
            src={abilityIcon(pick.ult_name)}
            className="h-6 w-6 rounded-sm border border-accent-ult"
            alt={pick.ult_name}
            title={pick.ult_name}
            onError={(e) => (e.currentTarget.style.visibility = "hidden")}
          />
        ) : (
          <div className="h-6 w-6 rounded-sm border border-accent-ult bg-bg-cell" />
        )}
      </div>
    </div>
  );
}
