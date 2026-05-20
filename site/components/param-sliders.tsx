"use client";
import { useState } from "react";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
import type { Param } from "@/lib/types";

declare global {
  interface Window {
    loadPyodide: (opts: { indexURL: string }) => Promise<any>;
    pyodide: any;
  }
}

async function ensurePyodide() {
  if (window.pyodide) return window.pyodide;
  await new Promise<void>((resolve, reject) => {
    const s = document.createElement("script");
    s.src = "https://cdn.jsdelivr.net/pyodide/v0.26.0/full/pyodide.js";
    s.onload = () => resolve();
    s.onerror = reject;
    document.head.appendChild(s);
  });
  window.pyodide = await window.loadPyodide({
    indexURL: "https://cdn.jsdelivr.net/pyodide/v0.26.0/full/",
  });
  await window.pyodide.loadPackage(["torch_activation"]).catch(() => {});
  return window.pyodide;
}

export function ParamSliders({
  activationName,
  params,
}: {
  activationName: string;
  params: Param[];
}) {
  const [values, setValues] = useState<Record<string, number>>(
    Object.fromEntries(params.map((p) => [p.name, p.default as number]))
  );
  const [loading, setLoading] = useState(false);
  const [plotSvg, setPlotSvg] = useState<string | null>(null);
  const [activated, setActivated] = useState(false);

  async function runPlot(vals: Record<string, number>) {
    setLoading(true);
    try {
      const py = await ensurePyodide();
      const kwargs = Object.entries(vals)
        .map(([k, v]) => `${k}=${v}`)
        .join(", ");
      const code = `
import torch, torch_activation, io, base64
import plotly.io as pio
import plotly.graph_objects as go

x = torch.linspace(-5, 5, 200)
x.requires_grad_(True)
m = torch_activation.${activationName}(${kwargs})
y = m(x)
d = torch.autograd.grad(y, x, torch.ones_like(y))[0]

fig = go.Figure()
fig.add_trace(go.Scatter(x=x.detach().tolist(), y=y.detach().tolist(), name="${activationName}"))
fig.add_trace(go.Scatter(x=x.detach().tolist(), y=d.detach().tolist(), name="derivative", line=dict(dash="dot")))
pio.to_json(fig)
      `;
      const figJson = await py.runPythonAsync(code);
      setPlotSvg(figJson);
    } catch (e) {
      console.error(e);
    }
    setLoading(false);
  }

  if (params.length === 0) return null;

  if (!activated) {
    return (
      <Button variant="outline" onClick={() => { setActivated(true); runPlot(values); }}>
        Enable interactive parameters
      </Button>
    );
  }

  return (
    <div className="space-y-4">
      {params.map((p) => (
        <div key={p.name} className="space-y-1">
          <div className="flex justify-between text-sm">
            <span>{p.name}</span>
            <span className="text-muted-foreground">{values[p.name]}</span>
          </div>
          <Slider
            min={-5}
            max={5}
            step={p.type === "int" ? 1 : 0.1}
            value={[values[p.name]] as [number]}
            onValueChange={(v) => {
              const val = Array.isArray(v) ? v[0] : v;
              const next = { ...values, [p.name]: val };
              setValues(next);
              runPlot(next);
            }}
          />
        </div>
      ))}
      {loading && <p className="text-xs text-muted-foreground">Computing...</p>}
    </div>
  );
}
