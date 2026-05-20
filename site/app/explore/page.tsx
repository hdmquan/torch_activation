"use client";
import { useEffect, useState } from "react";
import { Sidebar } from "@/components/sidebar";
import { ActivationPanel } from "@/components/activation-panel";
import { loadData } from "@/lib/data";
import type { SiteData } from "@/lib/types";

export default function ExplorePage() {
  const [data, setData] = useState<SiteData | null>(null);
  const [selected, setSelected] = useState<string | null>(null);

  useEffect(() => {
    loadData().then((d) => {
      setData(d);
      setSelected(d.activations[0]?.name ?? null);
    });
  }, []);

  if (!data) return <div className="flex h-[calc(100vh-57px)] items-center justify-center text-muted-foreground">Loading...</div>;

  const activation = data.activations.find((a) => a.name === selected);

  return (
    <div className="flex h-[calc(100vh-57px)]">
      <aside className="w-64 shrink-0 border-r">
        <Sidebar data={data} selected={selected} onSelect={setSelected} />
      </aside>
      <main className="flex-1 overflow-auto">
        {activation ? (
          <ActivationPanel activation={activation} data={data} />
        ) : (
          <div className="flex h-full items-center justify-center text-muted-foreground">
            Select an activation
          </div>
        )}
      </main>
    </div>
  );
}
