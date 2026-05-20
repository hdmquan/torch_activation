import type { Activation, Family } from "./types";

export function getHeadline(family: Family, activations: Activation[]): Activation[] {
  const names = new Set(family.headline);
  const head = activations.filter((a) => names.has(a.name) && a.family === family.label);
  if (head.length >= 4) return head.slice(0, 4);
  const rest = activations.filter((a) => a.family === family.label && !names.has(a.name));
  return [...head, ...rest].slice(0, 4);
}

export function getAll(family: Family, activations: Activation[]): Activation[] {
  return activations.filter((a) => a.family === family.label);
}

export function slugify(label: string): string {
  return label.toLowerCase().replace(/[\s/]+/g, "-").replace(/-+/g, "-");
}
