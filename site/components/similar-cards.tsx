import Link from "next/link";
import { ArrowUpRight } from "lucide-react";
import type { Activation } from "@/lib/types";

export function SimilarCards({ activations }: { activations: Activation[] }) {
  if (activations.length === 0) return null;
  return (
    <section className="space-y-3">
      <h2 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
        Similar activations
      </h2>
      <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3">
        {activations.map((a) => (
          <Link
            key={a.name}
            href={`/${a.name}`}
            className="group flex items-start justify-between rounded-lg border bg-card px-3 py-2.5 transition-colors hover:border-foreground/30 hover:bg-accent/40"
          >
            <div className="min-w-0 flex-1">
              <p className="truncate font-mono text-sm font-medium">{a.name}</p>
              {a.tags.length > 0 && (
                <p className="mt-0.5 truncate text-[11px] text-muted-foreground">
                  {a.tags.slice(0, 3).join(" · ")}
                </p>
              )}
            </div>
            <ArrowUpRight className="ml-2 mt-0.5 h-3.5 w-3.5 shrink-0 text-muted-foreground transition-colors group-hover:text-foreground" />
          </Link>
        ))}
      </div>
    </section>
  );
}
