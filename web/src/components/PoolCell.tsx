import { useState } from "react";
import {
  useAddAbility,
  useAddHero,
  usePick,
  useRemoveAbility,
  useRemoveHero,
} from "@/hooks/useDraft";
import type { PickType } from "@/api/client";
import { abilityIcon, heroIcon } from "@/lib/icons";

type CellKind = "hero" | "basic" | "ult";

interface FilledCellProps {
  kind: CellKind;
  id: number;
  name: string;
  picked: boolean;
  rank?: number;
}

interface EmptyCellProps {
  kind: CellKind;
  slotIndex: number;
  datalistId: string;
}

interface PoolCellProps {
  kind: CellKind;
  id: number | null;
  name: string | null;
  isPicked: boolean;
  slotIndex: number;
  datalistId: string;
  rank?: number;
}

export function PoolCell(props: PoolCellProps) {
  if (props.id == null || props.name == null) {
    return (
      <EmptyCell
        kind={props.kind}
        slotIndex={props.slotIndex}
        datalistId={props.datalistId}
      />
    );
  }
  return (
    <FilledCell
      kind={props.kind}
      id={props.id}
      name={props.name}
      picked={props.isPicked}
      rank={props.rank}
    />
  );
}

function FilledCell({ kind, id, name, picked, rank }: FilledCellProps) {
  const pick = usePick();
  const removeHero = useRemoveHero();
  const removeAbility = useRemoveAbility();

  const isHero = kind === "hero";
  const sizeClass = isHero ? "h-[46px] w-[72px]" : "h-[46px] w-[46px]";
  const iconSize = isHero ? "h-[42px] w-[68px]" : "h-[42px] w-[42px]";
  const icon = isHero ? heroIcon(name) : abilityIcon(name);

  const onPick = () => {
    if (picked) return;
    pick.mutate({ id, type: kind as PickType, is_random: false });
  };

  const onRemove = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (isHero) removeHero.mutate(name);
    else removeAbility.mutate(name);
  };

  return (
    <div
      className={`${sizeClass} relative flex items-center gap-1 overflow-hidden rounded bg-bg-cell p-0.5 ${
        picked ? "opacity-20" : "cursor-pointer hover:brightness-125"
      }`}
      onClick={onPick}
    >
      {rank != null && (
        <span className="absolute top-0 left-0 z-1 rounded-tl-[2px] rounded-br-[2px] bg-black/80 px-0.5 text-[8px] font-bold leading-tight text-accent">
          {rank}
        </span>
      )}
      {!picked && (
        <button
          className="absolute top-0 right-0 z-1 rounded-bl-[3px] bg-black/50 px-0.5 leading-[1.4] text-[10px] text-accent-error hover:bg-accent-error hover:text-white"
          onClick={onRemove}
        >
          ×
        </button>
      )}
      <img
        src={icon}
        alt={name}
        title={name}
        className={`${iconSize} flex-shrink-0 rounded-sm object-cover`}
        onError={(e) => (e.currentTarget.style.visibility = "hidden")}
      />
    </div>
  );
}

function EmptyCell({ kind, slotIndex, datalistId }: EmptyCellProps) {
  const [val, setVal] = useState("");
  const addHero = useAddHero();
  const addAbility = useAddAbility();

  const submit = () => {
    const name = val.trim();
    if (!name) return;
    if (kind === "hero") {
      addHero.mutate({ name, slot: slotIndex });
    } else {
      addAbility.mutate({ name, slot: slotIndex, kind });
    }
    setVal("");
  };

  const sizeClass = kind === "hero" ? "h-[46px] w-[72px]" : "h-[46px] w-[46px]";

  return (
    <div className={`${sizeClass} relative rounded border border-dashed border-text-very-dim`}>
      <input
        list={datalistId}
        value={val}
        onChange={(e) => setVal(e.target.value)}
        onBlur={submit}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            submit();
            e.preventDefault();
          }
        }}
        placeholder="+"
        className="absolute inset-0 h-full w-full bg-transparent p-0.5 text-center text-[8px] text-text-muted outline-none placeholder:text-[14px] placeholder:text-text-very-dim"
      />
    </div>
  );
}
