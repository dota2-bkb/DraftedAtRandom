import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  api,
  type AbilityEntry,
  type AddAbilityBody,
  type AddHeroBody,
  type DraftEnvelope,
  type HeroEntry,
  type PickBody,
  type SetPreferencesBody,
} from "@/api/client";

const DRAFT_KEY = ["draft"] as const;
const HEROES_KEY = ["lookups", "heroes"] as const;
const ABILITIES_KEY = ["lookups", "abilities"] as const;

export function useDraft() {
  return useQuery<DraftEnvelope>({ queryKey: DRAFT_KEY, queryFn: api.getDraft });
}

export function useHeroes() {
  return useQuery<HeroEntry[]>({
    queryKey: HEROES_KEY,
    queryFn: api.heroes,
    staleTime: Infinity,
  });
}

export function useAbilities() {
  return useQuery<AbilityEntry[]>({
    queryKey: ABILITIES_KEY,
    queryFn: api.abilities,
    staleTime: Infinity,
  });
}

function useDraftMutation<TVars>(fn: (vars: TVars) => Promise<DraftEnvelope>) {
  const qc = useQueryClient();
  return useMutation<DraftEnvelope, Error, TVars>({
    mutationFn: fn,
    onSuccess: (data) => qc.setQueryData(DRAFT_KEY, data),
  });
}

export function useReset() {
  return useDraftMutation<void>(() => api.reset());
}

export function useUndo() {
  return useDraftMutation<void>(() => api.undo());
}

export function usePick() {
  return useDraftMutation<PickBody>((body) => api.pick(body));
}

export function useAddHero() {
  return useDraftMutation<AddHeroBody>((body) => api.addHero(body));
}

export function useAddAbility() {
  return useDraftMutation<AddAbilityBody>((body) => api.addAbility(body));
}

export function useRemoveHero() {
  return useDraftMutation<string>((name) => api.removeHero(name));
}

export function useRemoveAbility() {
  return useDraftMutation<string>((name) => api.removeAbility(name));
}

export function useSetPreset() {
  return useDraftMutation<SetPreferencesBody>((body) => api.setPreset(body));
}
