"use client";
import { useState } from "react";
import { Button } from "@/components/ui/button";

export function CodeSnippet({ name }: { name: string }) {
  const [copied, setCopied] = useState(false);
  const code = `import torch\nimport torch_activation\n\nm = torch_activation.${name}()\nx = torch.randn(2, 3)\noutput = m(x)`;

  function copy() {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }

  return (
    <div className="relative rounded-md bg-muted p-4 font-mono text-sm">
      <pre>{code}</pre>
      <Button
        size="sm"
        variant="ghost"
        className="absolute right-2 top-2 h-7 text-xs"
        onClick={copy}
      >
        {copied ? "Copied" : "Copy"}
      </Button>
    </div>
  );
}
