"use client";
import { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { SearchBar } from "./search-bar";
import { getHeadline, getAll } from "@/lib/families";
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
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  function toggleTag(tag: Tag) {
    setActiveTags((prev) =>
      prev.includes(tag) ? prev.filter((t) => t !== tag) : [...prev, tag]
    );
  }

  function toggleExpand(id: string) {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  const filtered = filterActivations(data, activeTags, query);
  const filteredNames = new Set(filtered.map((a) => a.name));

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
        <div className="space-y-5 p-3 pr-2">
          {data.families.map((family) => {
            const all = getAll(family, data.activations).filter((a) =>
              filteredNames.has(a.name)
            );
            if (all.length === 0) return null;
            const headline = getHeadline(family, filtered);
            const isExpanded = expanded.has(family.id);
            const shown = isExpanded ? all : headline;
            const hasMore = all.length > headline.length;

            return (
              <div key={family.id}>
                <div className="mb-1.5 flex items-center justify-between px-2">
                  <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                    {family.label}
                  </p>
                  <Badge
                    variant="secondary"
                    className="h-4 px-1.5 text-[10px] font-normal"
                  >
                    {all.length}
                  </Badge>
                </div>
                <div className="space-y-px">
                  {shown.map((act) => {
                    const isSelected = selected === act.name;
                    return (
                      <button
                        key={act.name}
                        onClick={() => onSelect(act.name)}
                        className={`group relative flex w-full items-center rounded-md py-1.5 pl-3 pr-2 text-left font-mono text-[13px] transition-colors ${
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
                    );
                  })}
                  {hasMore && (
                    <button
                      onClick={() => toggleExpand(family.id)}
                      className="flex w-full items-center gap-1 rounded-md px-3 py-1.5 text-left text-[11px] text-muted-foreground transition-colors hover:bg-accent/40 hover:text-foreground"
                    >
                      {isExpanded ? (
                        <ChevronDown className="h-3 w-3" />
                      ) : (
                        <ChevronRight className="h-3 w-3" />
                      )}
                      {isExpanded ? "Show less" : `${all.length - headline.length} more`}
                    </button>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </ScrollArea>
    </div>
  );
}
