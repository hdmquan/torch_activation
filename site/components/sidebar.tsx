"use client";
import { useState } from "react";
import Link from "next/link";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
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
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  }

  const filtered = filterActivations(data, activeTags, query);
  const filteredNames = new Set(filtered.map((a) => a.name));

  return (
    <div className="flex h-full flex-col gap-3 p-3">
      <SearchBar value={query} onChange={setQuery} />

      <div className="flex flex-wrap gap-1">
        {ALL_TAGS.map((tag) => (
          <Badge
            key={tag}
            variant={activeTags.includes(tag) ? "default" : "outline"}
            className="cursor-pointer text-xs"
            onClick={() => toggleTag(tag)}
          >
            {tag}
          </Badge>
        ))}
      </div>

      <Separator />

      <ScrollArea className="flex-1">
        <div className="space-y-4 pr-2">
          {data.families.map((family) => {
            const all = getAll(family, data.activations).filter((a) =>
              filteredNames.has(a.name)
            );
            if (all.length === 0) return null;
            const headline = getHeadline(family, filtered);
            const isExpanded = expanded.has(family.id);
            const shown = isExpanded ? all : headline;

            return (
              <div key={family.id}>
                <p className="mb-1 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                  {family.label}
                </p>
                <div className="space-y-0.5">
                  {shown.map((act) => (
                    <button
                      key={act.name}
                      onClick={() => onSelect(act.name)}
                      className={`w-full rounded px-2 py-1 text-left text-sm transition-colors hover:bg-accent ${
                        selected === act.name ? "bg-accent font-medium" : ""
                      }`}
                    >
                      {act.name}
                    </button>
                  ))}
                  {all.length > headline.length && (
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-6 w-full text-xs text-muted-foreground"
                      onClick={() => toggleExpand(family.id)}
                    >
                      {isExpanded ? "Show less" : `Show all ${all.length}`}
                    </Button>
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
