"use client";
import "katex/dist/katex.min.css";
import { InlineMath, BlockMath } from "react-katex";

export function Formula({ latex, block = false }: { latex: string; block?: boolean }) {
  if (!latex) return null;
  try {
    return block ? <BlockMath math={latex} /> : <InlineMath math={latex} />;
  } catch {
    return <code className="text-sm">{latex}</code>;
  }
}
