"use client";
import { useState } from "react";
import { Button } from "@/components/ui/button";

export function PlotView({ src, name }: { src: string; name: string }) {
  const [showDerivative, setShowDerivative] = useState(true);
  const base = process.env.NEXT_PUBLIC_BASE_PATH || "";

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <Button
          size="sm"
          variant={showDerivative ? "default" : "outline"}
          onClick={() => setShowDerivative((v) => !v)}
        >
          Derivative
        </Button>
      </div>
      <div className="relative overflow-hidden rounded-lg border bg-background">
        <img
          src={`${base}${src}`}
          alt={`${name} activation plot`}
          className="w-full"
          onError={(e) => {
            (e.target as HTMLImageElement).style.display = "none";
          }}
        />
      </div>
    </div>
  );
}
