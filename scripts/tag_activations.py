import sys

import torch

import torch_activation as ta


def _tag(name: str, cls) -> list[str]:
    tags = []
    try:
        m = cls()
        x = torch.linspace(-100, 100, 500)
        with torch.no_grad():
            y = m(x)
        if not (torch.isnan(y).any() or torch.isinf(y).any()):
            tags.append("overflow-safe")
        diffs = y[1:] - y[:-1]
        if (diffs >= 0).all() or (diffs <= 0).all():
            tags.append("monotonic")
        y_range = y.max().item() - y.min().item()
        if y_range < 5:
            tags.append("bounded")
        x64 = torch.linspace(-3, 3, 8, dtype=torch.float64).requires_grad_(True)
        try:
            torch.autograd.gradcheck(m, (x64,), eps=1e-6, atol=1e-4, raise_exception=True)
            tags.append("smooth")
        except Exception:
            pass
    except Exception as e:
        print(f"[tag] {name}: {e}", file=sys.stderr)
    return tags


def tag_all() -> dict[str, list[str]]:
    return {name: _tag(name, ta._ACTIVATIONS[name]["class"]) for name in ta.get_all_activations()}


if __name__ == "__main__":
    import json

    json.dump(tag_all(), sys.stdout, indent=2)
