import type { Activation, Family, SiteData } from "./types";

let _cache: SiteData | null = null;

export async function loadData(): Promise<SiteData> {
  if (_cache) return _cache;
  const base = process.env.NEXT_PUBLIC_BASE_PATH || "";
  const res = await fetch(`${base}/data.json`);
  _cache = await res.json();
  return _cache!;
}

export function getActivation(data: SiteData, name: string): Activation | undefined {
  return data.activations.find((a) => a.name === name);
}

export function getFamilies(data: SiteData): Family[] {
  return data.families;
}

export function filterActivations(
  data: SiteData,
  tags: string[],
  query: string
): Activation[] {
  return data.activations.filter((a) => {
    if (tags.length > 0 && !tags.every((t) => a.tags.includes(t as any))) return false;
    if (query && !a.name.toLowerCase().includes(query.toLowerCase())) return false;
    return true;
  });
}
