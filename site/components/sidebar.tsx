"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { SearchBar } from "./search-bar";
import { getAll } from "@/lib/families";
import { filterActivations } from "@/lib/data";
import type { SiteData, Tag } from "@/lib/types";

const ALL_TAGS: Tag[] = ["smooth", "monotonic", "bounded", "overflow-safe"];

export function Sidebar({
  data,
  selected,
  onSelect,
}: {
  data: SiteData;
  selected: string | null;
  onSelect: (name: string) => void;
}) {
  const [activeTags, setActiveTags] = useState<Tag[]>([]);
  const [query, setQuery] = useState("");
  const selectedRef = useRef<HTMLButtonElement>(null);

  function toggleTag(tag: Tag) {
    setActiveTags((prev) =>
      prev.includes(tag) ? prev.filter((t) => t !== tag) : [...prev, tag]
    );
  }

  const filtered = filterActivations(data, activeTags, query);
  const filteredNames = useMemo(() => new Set(filtered.map((a) => a.name)), [filtered]);

  const groups = useMemo(
    () =>
      data.families
        .map((family) => ({
          family,
          items: getAll(family, data.activations).filter((a) =>
            filteredNames.has(a.name)
          ),
        }))
        .filter((g) => g.items.length > 0),
    [data, filteredNames]
  );

  // Scroll selected row into view on initial render.
  useEffect(() => {
    if (selectedRef.current) {
      selectedRef.current.scrollIntoView({ block: "center", behavior: "auto" });
    }
    // Only run once on mount; subsequent selections happen on user click and
    // don't need to recenter the sidebar.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="flex h-full flex-col bg-sidebar">
      <div className="space-y-3 border-b px-3 py-3">
        <SearchBar value={query} onChange={setQuery} />

        <div className="flex flex-wrap gap-1.5">
          {ALL_TAGS.map((tag) => {
            const active = activeTags.includes(tag);
            return (
              <button
                key={tag}
                onClick={() => toggleTag(tag)}
                className={`cursor-pointer rounded-full px-2.5 py-0.5 text-[11px] font-medium transition-colors ${
                  active
                    ? "bg-foreground text-background"
                    : "border border-border bg-transparent text-muted-foreground hover:border-foreground/40 hover:text-foreground"
                }`}
              >
                {tag}
              </button>
            );
          })}
        </div>

        <p className="text-[11px] text-muted-foreground">
          {filtered.length} of {data.activations.length} activations
        </p>
      </div>

      <ScrollArea className="min-h-0 flex-1 overscroll-contain">
        <div className="pb-6">
          {groups.length === 0 ? (
            <p className="px-4 py-8 text-center text-sm text-muted-foreground">
              No matches
            </p>
          ) : (
            groups.map(({ family, items }) => (
              <section key={family.id}>
                <div className="sticky top-0 z-10 flex items-center justify-between border-b border-border/60 bg-sidebar/95 px-3 py-1.5 backdrop-blur supports-[backdrop-filter]:bg-sidebar/80">
                  <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                    {family.label}
                  </p>
                  <span className="font-mono text-[10px] text-muted-foreground/70">
                    {items.length}
                  </span>
                </div>
                <ul className="space-y-px px-2 py-1.5">
                  {items.map((act) => {
                    const isSelected = selected === act.name;
                    return (
                      <li key={act.name}>
                        <button
                          ref={isSelected ? selectedRef : null}
                          onClick={() => onSelect(act.name)}
                          className={`relative flex w-full items-center rounded-md py-1 pl-3 pr-2 text-left font-mono text-[13px] transition-colors ${
                            isSelected
                              ? "bg-accent text-accent-foreground"
                              : "text-foreground/80 hover:bg-accent/50 hover:text-foreground"
                          }`}
                        >
                          {isSelected && (
                            <span className="absolute left-0 top-1/2 h-4 w-0.5 -translate-y-1/2 rounded-full bg-foreground" />
                          )}
                          {act.name}
                        </button>
                      </li>
                    );
                  })}
                </ul>
              </section>
            ))
          )}
        </div>
      </ScrollArea>
    </div>
  );
}
