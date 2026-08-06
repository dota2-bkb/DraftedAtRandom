import { useMemo } from "react";
import type { AnnotatedDraft, SuggestionEntry } from "@/api/client";
import { useAbilities, useHeroes } from "@/hooks/useDraft";
import { PoolCell } from "./PoolCell";

interface PoolProps {
  state: AnnotatedDraft;
  suggestions: SuggestionEntry[];
}

export function Pool({ state, suggestions }: PoolProps) {
  const { data: heroes } = useHeroes();
  const { data: abilities } = useAbilities();

  const heroOpts = useMemo(
    () =>
      (heroes ?? []).filter((h) => !state.hero_pool_all.includes(h.id)).map((h) => h.name),
    [heroes, state.hero_pool_all],
  );
  const basicOpts = useMemo(
    () =>
      (abilities ?? [])
        .filter((a) => !a.is_ult && !state.basic_pool_all.includes(a.id))
        .map((a) => a.name),
    [abilities, state.basic_pool_all],
  );
  const ultOpts = useMemo(
    () =>
      (abilities ?? [])
        .filter((a) => a.is_ult && !state.ult_pool_all.includes(a.id))
        .map((a) => a.name),
    [abilities, state.ult_pool_all],
  );

  // Suggestion rank lookup, sorted by prob desc
  const rankMap = useMemo(() => {
    const sorted = [...suggestions].sort((a, b) => b.prob - a.prob);
    const m = new Map<string, number>();
    sorted.forEach((sg, i) => m.set(`${sg.type}:${sg.id}`, i + 1));
    return m;
  }, [suggestions]);

  const heroRemaining = new Set(state.hero_pool_remaining);
  const basicRemaining = new Set(state.basic_pool_remaining);
  const ultRemaining = new Set(state.ult_pool_remaining);

  // Layout:
  // Left half: heroes 0..5, ult grid rows [[0,1,2],[3,4,5]]
  // Right half: heroes 6..11 (mirrored), ult grid [[8,7,6],[11,10,9]]
  return (
    <div className="flex flex-shrink-0 justify-center gap-2 rounded-md bg-bg-card p-1.5">
      <PoolHalf
        heroIndices={[0, 1, 2, 3, 4, 5]}
        ultLayout={[[0, 1, 2], [3, 4, 5]]}
        state={state}
        heroRemaining={heroRemaining}
        basicRemaining={basicRemaining}
        ultRemaining={ultRemaining}
        rankMap={rankMap}
        align="end"
      />
      <PoolHalf
        heroIndices={[6, 7, 8, 9, 10, 11]}
        ultLayout={[[8, 7, 6], [11, 10, 9]]}
        state={state}
        heroRemaining={heroRemaining}
        basicRemaining={basicRemaining}
        ultRemaining={ultRemaining}
        rankMap={rankMap}
        mirrored
      />

      <datalist id="hero-list">
        {heroOpts.map((n) => (
          <option key={n} value={n} />
        ))}
      </datalist>
      <datalist id="ability-list">
        {basicOpts.map((n) => (
          <option key={n} value={n} />
        ))}
      </datalist>
      <datalist id="ult-list">
        {ultOpts.map((n) => (
          <option key={n} value={n} />
        ))}
      </datalist>
    </div>
  );
}

interface PoolHalfProps {
  heroIndices: number[];
  ultLayout: number[][];
  state: AnnotatedDraft;
  heroRemaining: Set<number>;
  basicRemaining: Set<number>;
  ultRemaining: Set<number>;
  rankMap: Map<string, number>;
  mirrored?: boolean;
  align?: "start" | "end";
}

function PoolHalf({
  heroIndices,
  ultLayout,
  state,
  heroRemaining,
  basicRemaining,
  ultRemaining,
  rankMap,
  mirrored = false,
  align,
}: PoolHalfProps) {
  return (
    <div className={`flex flex-col ${align === "end" ? "items-end" : ""}`}>
      <div className="grid grid-cols-3 gap-[3px] mb-1.5" style={{ gridTemplateColumns: "repeat(3, 46px)" }}>
        {ultLayout.flatMap((row) =>
          row.map((ui) => {
            const id = state.ult_pool_all[ui] ?? null;
            const name = state.ult_pool_all_names[ui] ?? null;
            const rank = id != null ? rankMap.get(`ult:${id}`) : undefined;
            return (
              <PoolCell
                key={`ult-${ui}`}
                kind="ult"
                id={id}
                name={name}
                isPicked={id != null && !ultRemaining.has(id)}
                slotIndex={ui}
                datalistId="ult-list"
                rank={rank}
              />
            );
          }),
        )}
      </div>

      {heroIndices.map((hi) => {
        const heroId = state.hero_pool_all[hi] ?? null;
        const heroName = state.hero_pool_all_names[hi] ?? null;
        const heroRank = heroId != null ? rankMap.get(`hero:${heroId}`) : undefined;

        const heroCell = (
          <PoolCell
            key={`hero-${hi}`}
            kind="hero"
            id={heroId}
            name={heroName}
            isPicked={heroId != null && !heroRemaining.has(heroId)}
            slotIndex={hi}
            datalistId="hero-list"
            rank={heroRank}
          />
        );

        const basicCells = [0, 1, 2].map((j) => {
          const bi = hi * 3 + j;
          const id = state.basic_pool_all[bi] ?? null;
          const name = state.basic_pool_all_names[bi] ?? null;
          const rank = id != null ? rankMap.get(`basic:${id}`) : undefined;
          return (
            <PoolCell
              key={`basic-${bi}`}
              kind="basic"
              id={id}
              name={name}
              isPicked={id != null && !basicRemaining.has(id)}
              slotIndex={bi}
              datalistId="ability-list"
              rank={rank}
            />
          );
        });

        return (
          <div
            key={`row-${hi}`}
            className="grid gap-[3px] mb-[3px]"
            style={{
              gridTemplateColumns: mirrored ? "repeat(3, 46px) 72px" : "72px repeat(3, 46px)",
            }}
          >
            {mirrored ? (
              <>
                {basicCells}
                {heroCell}
              </>
            ) : (
              <>
                {heroCell}
                {basicCells}
              </>
            )}
          </div>
        );
      })}
    </div>
  );
}
