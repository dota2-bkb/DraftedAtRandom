import type { components, paths } from "./types";

export type PresetEntry = components["schemas"]["PresetEntry"];
export type PreferencesState = components["schemas"]["PreferencesState"];

// The recommender-selection surface is not in the generated OpenAPI schema;
// its fields are typed here.
type AutoDraftEnvelope = components["schemas"]["DraftEnvelope"];
export type DraftEnvelope = AutoDraftEnvelope & {
  recommenders?: string[];   // "bc" + loaded vector-Q variants (+ "reweight_bc")
  recommender?: string;      // the active one
  reweight_beta?: number;    // current reweight-BC tilt strength β
};
export type AnnotatedDraft = components["schemas"]["AnnotatedDraft"];
export type AnnotatedPlayerPick = components["schemas"]["AnnotatedPlayerPick"];
export type SuggestionEntry = components["schemas"]["SuggestionEntry"];
export type ConfidenceInfo = components["schemas"]["ConfidenceInfo"];
export type HeroEntry = components["schemas"]["HeroEntry"];
export type AbilityEntry = components["schemas"]["AbilityEntry"];
export type AddHeroBody = components["schemas"]["AddHeroBody"];
export type AddAbilityBody = components["schemas"]["AddAbilityBody"];
export type PickBody = components["schemas"]["PickBody"];
export interface SetPreferencesBody {
  // The generated SetPreferencesBody carries only a required `preset`; the
  // server also accepts recommender-selection updates, typed here.
  preset?: string;
  recommender?: string;   // "bc" | "q" | "trial" | "reweight_bc"
  reweight_beta?: number; // reweight-BC tilt strength β (β=0 ⇒ pure BC)
}

export type PickType = "hero" | "basic" | "ult";

class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(
  method: string,
  path: keyof paths | string,
  body?: unknown,
): Promise<T> {
  const res = await fetch(path, {
    method,
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  const text = await res.text();
  const data = text ? JSON.parse(text) : null;
  if (!res.ok) {
    const msg = data?.error ?? data?.detail ?? res.statusText;
    throw new ApiError(res.status, msg);
  }
  return data as T;
}

export const api = {
  getDraft: () => request<DraftEnvelope>("GET", "/api/draft"),
  reset: () => request<DraftEnvelope>("POST", "/api/draft/reset"),
  undo: () => request<DraftEnvelope>("POST", "/api/draft/undo"),
  pick: (body: PickBody) => request<DraftEnvelope>("POST", "/api/draft/picks", body),
  addHero: (body: AddHeroBody) =>
    request<DraftEnvelope>("POST", "/api/draft/pool/heroes", body),
  addAbility: (body: AddAbilityBody) =>
    request<DraftEnvelope>("POST", "/api/draft/pool/abilities", body),
  removeHero: (name: string) =>
    request<DraftEnvelope>("DELETE", `/api/draft/pool/heroes/${encodeURIComponent(name)}`),
  removeAbility: (name: string) =>
    request<DraftEnvelope>("DELETE", `/api/draft/pool/abilities/${encodeURIComponent(name)}`),
  setPreset: (body: SetPreferencesBody) =>
    request<DraftEnvelope>("POST", "/api/draft/preferences", body),
  heroes: () => request<HeroEntry[]>("GET", "/api/lookups/heroes"),
  abilities: () => request<AbilityEntry[]>("GET", "/api/lookups/abilities"),
};

export { ApiError };
