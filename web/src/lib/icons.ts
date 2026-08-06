const CDN = "https://cdn.cloudflare.steamstatic.com/apps/dota2/images/dota_react";

// Canonical script name -> CDN image name, where Valve's react image
// deviates from the ability's script name.
const ABILITY_RENAMES: Record<string, string> = {
  mirana_moonlight_shadow: "mirana_invis",
};

export function heroIcon(key: string): string {
  return `${CDN}/heroes/${key}.png`;
}

export function abilityIcon(key: string): string {
  const k = key.replace(/_ad$/, "");
  return `${CDN}/abilities/${ABILITY_RENAMES[k] ?? k}.png`;
}
