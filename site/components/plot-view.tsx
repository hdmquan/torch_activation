"use client";

export function PlotView({ src, name }: { src: string; name: string }) {
  const base = process.env.NEXT_PUBLIC_BASE_PATH || "";

  return (
    <figure className="overflow-hidden rounded-xl border bg-card">
      <figcaption className="flex items-center justify-between border-b bg-muted/40 px-4 py-2">
        <span className="text-xs font-medium text-muted-foreground">
          f(x) and f&apos;(x)
        </span>
        <span className="font-mono text-[10px] text-muted-foreground">
          x ∈ [−5, 5]
        </span>
      </figcaption>
      <img
        src={`${base}${src}`}
        alt={`${name} activation and its derivative`}
        className="w-full bg-white"
        onError={(e) => {
          (e.target as HTMLImageElement).style.display = "none";
        }}
      />
    </figure>
  );
}
