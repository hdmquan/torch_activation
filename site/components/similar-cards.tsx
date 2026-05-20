import Link from "next/link";
import { Badge } from "@/components/ui/badge";
import type { Activation } from "@/lib/types";

export function SimilarCards({ activations }: { activations: Activation[] }) {
  if (activations.length === 0) return null;
  return (
    <div className="space-y-2">
      <p className="text-sm font-medium text-muted-foreground">Similar</p>
      <div className="flex flex-wrap gap-2">
        {activations.map((a) => (
          <Link key={a.name} href={`/${a.name}`}>
            <Badge variant="secondary" className="cursor-pointer hover:bg-secondary/80">
              {a.name}
            </Badge>
          </Link>
        ))}
      </div>
    </div>
  );
}
