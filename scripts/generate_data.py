import inspect
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch_activation as ta
from scripts.tag_activations import tag_all
from torch_activation.utils import plot_activation

MODULE_TO_FAMILY = {
    "torch_activation.classical.relu": "ReLU",
    "torch_activation.classical.sigmoid": "Sigmoid",
    "torch_activation.classical.sigmoid_weighted": "Sigmoid",
    "torch_activation.classical.softplus": "Softplus / Softmax",
    "torch_activation.classical.softmax": "Softplus / Softmax",
    "torch_activation.classical.trig": "Trigonometric",
    "torch_activation.classical.sqrt": "Power / Root",
    "torch_activation.classical.squared": "Power / Root",
    "torch_activation.classical.polynomial": "Polynomial / Exponential",
    "torch_activation.classical.exp": "Polynomial / Exponential",
    "torch_activation.classical.piecewise": "Piecewise / Other",
    "torch_activation.classical.other": "Piecewise / Other",
    "torch_activation.classical.maxsig": "Piecewise / Other",
    "torch_activation.classical.caf": "Chaotic",
    "torch_activation.classical.etanh": "Chaotic",
    "torch_activation.classical.glu": "Piecewise / Other",
    "torch_activation.classical.layer": "Learnable",
    "torch_activation.adaptive.relu": "Adaptive ReLU",
    "torch_activation.adaptive.faaf": "Adaptive ReLU",
    "torch_activation.adaptive.sigmoid": "Adaptive Sigmoid",
    "torch_activation.adaptive.sigmoid_weighted": "Adaptive Sigmoid",
    "torch_activation.adaptive.s_shaped": "Adaptive Sigmoid",
    "torch_activation.adaptive.melu": "Adaptive Other",
    "torch_activation.adaptive.abu": "Adaptive Other",
    "torch_activation.adaptive.msrf": "Adaptive Other",
    "torch_activation.adaptive.laaf": "Adaptive Other",
    "torch_activation.adaptive.other": "Adaptive Other",
    "torch_activation.classical.learnable": "Learnable",
}

HEADLINE = {
    "ReLU": ["ReLU", "LReLU", "ELU", "PReLU"],
    "Sigmoid": ["SiLU", "Mish", "GELU", "Swish"],
    "Softplus / Softmax": ["Softplus", "Softmax", "ParametricSoftplus", "SoftPlusPlus"],
    "Trigonometric": ["Sine", "Cosine", "GCU", "ASU"],
    "Power / Root": ["SquaredReLU", "ISRU", "SQNL", "SQRT"],
    "Polynomial / Exponential": ["NCU", "Polyexp", "Exponential", "Wave"],
    "Piecewise / Other": ["BentIdentity", "BiFiring", "SPOCU", "KDAC"],
    "Chaotic": ["FCAF_Hidden", "FCAF_Output", "CCAF", "ETanh"],
    "Adaptive ReLU": ["PReLU", "DPReLU", "SMU", "SAU"],
    "Adaptive Sigmoid": ["PELU", "PSwish", "SCMish", "TSwish"],
    "Adaptive Other": ["MeLU", "ShiLU", "StarReLU", "LAAF"],
    "Learnable": [],
}


def _extract_description(doc: str) -> str:
    for line in doc.strip().splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def _extract_formula(doc: str) -> str:
    m = re.search(r":math:`([^`]+)`", doc)
    if m:
        return m.group(1)
    m = re.search(r"\.\. math::\s*\n\s*(.+)", doc)
    if m:
        return m.group(1).strip()
    return ""


def _extract_paper_ref(doc: str) -> str:
    m = re.search(r"(https?://\S+|arXiv:\S+|arxiv\.\S+)", doc, re.IGNORECASE)
    return m.group(0) if m else ""


def _get_params(cls) -> list[dict]:
    sig = inspect.signature(cls.__init__)
    out = []
    for k, v in sig.parameters.items():
        if k in ("self", "inplace") or v.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if v.default is inspect.Parameter.empty:
            continue
        d = v.default
        if isinstance(d, bool):
            out.append({"name": k, "default": d, "type": "bool"})
        elif isinstance(d, int):
            out.append({"name": k, "default": d, "type": "int"})
        elif isinstance(d, float):
            out.append({"name": k, "default": d, "type": "float"})
    return out


def _similar(name: str, family: str, tags: list[str], all_acts: list[dict]) -> list[str]:
    same = [a for a in all_acts if a["family"] == family and a["name"] != name]
    scored = sorted(same, key=lambda a: -len(set(a["tags"]) & set(tags)))
    return [a["name"] for a in scored[:3]]


def main(plots_dir: str = "site/public/plots", out: str = "site/public/data.json"):
    tags_map = tag_all()
    names = ta.get_all_activations()
    seen_names: dict[str, int] = {}
    acts = []

    for name in names:
        entry = ta._ACTIVATIONS[name]
        cls = entry["class"]
        mod = cls.__module__
        family = MODULE_TO_FAMILY.get(mod, "Other")
        doc = inspect.getdoc(cls) or ""

        display_name = name
        if name in seen_names:
            seen_names[name] += 1
            display_name = f"{name} ({family})"
        else:
            seen_names[name] = 1

        try:
            plot_activation(cls, params={}, save_dir=plots_dir, fmt="svg")
        except Exception as e:
            print(f"Warning: plot failed for {name}: {e}", file=sys.stderr)

        acts.append({
            "name": display_name,
            "family": family,
            "module": mod,
            "description": _extract_description(doc),
            "formula": _extract_formula(doc),
            "params": _get_params(cls),
            "tags": tags_map.get(name, []),
            "paper_ref": _extract_paper_ref(doc),
            "plot": f"/plots/{name}.svg",
            "similar": [],
        })

    for act in acts:
        act["similar"] = _similar(act["name"], act["family"], act["tags"], acts)

    families = []
    for label, headlines in HEADLINE.items():
        slug = label.lower().replace(" ", "-").replace("/", "").replace("--", "-")
        families.append({"id": slug, "label": label, "headline": headlines})

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"activations": acts, "families": families}, f, indent=2)
    print(f"Written {len(acts)} activations to {out}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--plots-dir", default="site/public/plots")
    p.add_argument("--out", default="site/public/data.json")
    args = p.parse_args()
    main(args.plots_dir, args.out)
