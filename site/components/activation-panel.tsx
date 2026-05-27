import { ExternalLink } from "lucide-react";
import { Formula } from "./formula";
import { PlotView } from "./plot-view";
import { CodeSnippet } from "./code-snippet";
import { SimilarCards } from "./similar-cards";
import { ParamSliders } from "./param-sliders";
import type { Activation, SiteData } from "@/lib/types";
import { getActivation } from "@/lib/data";

export function ActivationPanel({
  activation,
  data,
}: {
  activation: Activation;
  data: SiteData;
}) {
  const similarActs = activation.similar
    .map((n) => getActivation(data, n))
    .filter(Boolean) as Activation[];

  const refHref = activation.paper_ref
    ? activation.paper_ref.startsWith("http")
      ? activation.paper_ref
      : `https://arxiv.org/abs/${activation.paper_ref}`
    : null;

  return (
    <div className="mx-auto max-w-4xl space-y-8 px-6 py-8 lg:px-10 lg:py-10">
      <header className="space-y-3">
        <h1 className="font-mono text-3xl font-semibold tracking-tight">
          {activation.name}
        </h1>
        {activation.description && (
          <p className="max-w-2xl text-[15px] leading-relaxed text-muted-foreground">
            {activation.description}
          </p>
        )}
        {(activation.tags.length > 0 || refHref) && (
          <div className="flex flex-wrap items-center gap-1.5 pt-1">
            {activation.tags.map((tag) => (
              <span
                key={tag}
                className="rounded-full border border-border bg-muted/40 px-2.5 py-0.5 text-[11px] font-medium text-muted-foreground"
              >
                {tag}
              </span>
            ))}
            {refHref && (
              <a
                href={refHref}
                target="_blank"
                rel="noreferrer"
                className="ml-1 inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-[11px] font-medium text-muted-foreground transition-colors hover:text-foreground"
              >
                <ExternalLink className="h-3 w-3" />
                Paper
              </a>
            )}
          </div>
        )}
      </header>

      {activation.formula && (
        <section className="rounded-xl border bg-card px-6 py-5">
          <p className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
            Definition
          </p>
          <Formula latex={activation.formula} block />
        </section>
      )}

      <PlotView src={activation.plot} name={activation.name} />

      {activation.params.length > 0 && (
        <section className="space-y-3">
          <h2 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Parameters
          </h2>
          <ParamSliders
            activationName={activation.name}
            params={activation.params}
          />
        </section>
      )}

      <section className="space-y-3">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Usage
        </h2>
        <CodeSnippet name={activation.name} />
      </section>

      <SimilarCards activations={similarActs} />
    </div>
  );
}
