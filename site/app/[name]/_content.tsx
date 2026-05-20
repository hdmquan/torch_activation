"use client";
import { useEffect, useState } from "react";
import { ActivationPanel } from "@/components/activation-panel";
import { loadData } from "@/lib/data";
import type { SiteData } from "@/lib/types";

export function ActivationContent({ name }: { name: string }) {
  const [data, setData] = useState<SiteData | null>(null);

  useEffect(() => { loadData().then(setData); }, []);

  if (!data) return <div className="p-6 text-muted-foreground">Loading...</div>;
  const activation = data.activations.find((a) => a.name === name);
  if (!activation) return <div className="p-6 text-muted-foreground">Not found: {name}</div>;

  return <ActivationPanel activation={activation} data={data} />;
}
