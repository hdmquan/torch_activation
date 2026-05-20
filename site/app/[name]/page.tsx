import { Suspense } from "react";
import { ActivationContent } from "./_content";

export const dynamic = "force-static";

export function generateStaticParams() {
  return [{ name: "__placeholder__" }];
}

export default function ActivationPage({ params }: { params: { name: string } }) {
  return (
    <Suspense fallback={<div className="p-6 text-muted-foreground">Loading...</div>}>
      <ActivationContent name={params.name} />
    </Suspense>
  );
}
