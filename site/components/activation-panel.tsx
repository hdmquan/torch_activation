import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
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

  return (
    <div className="space-y-6 p-6">
      <div>
        <h1 className="text-2xl font-bold">{activation.name}</h1>
        <p className="mt-1 text-muted-foreground">{activation.description}</p>
      </div>

      {activation.formula && (
        <div className="rounded-lg border bg-muted/40 p-4">
          <Formula latex={activation.formula} block />
        </div>
      )}

      <PlotView src={activation.plot} name={activation.name} />

      {activation.params.length > 0 && (
        <ParamSliders
          activationName={activation.name}
          params={activation.params}
        />
      )}

      <div className="flex flex-wrap gap-2">
        {activation.tags.map((tag) => (
          <Badge key={tag} variant="outline">
            {tag}
          </Badge>
        ))}
      </div>

      <Separator />

      <CodeSnippet name={activation.name} />

      {activation.paper_ref && (
        <p className="text-sm text-muted-foreground">
          Reference:{" "}
          <a
            href={activation.paper_ref.startsWith("http") ? activation.paper_ref : `https://arxiv.org/abs/${activation.paper_ref}`}
            className="underline hover:text-foreground"
            target="_blank"
            rel="noreferrer"
          >
            {activation.paper_ref}
          </a>
        </p>
      )}

      <SimilarCards activations={similarActs} />
    </div>
  );
}
