import argparse

import torch_activation as ta
from torch_activation.utils import plot_activation

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", default="images/activation_images")
parser.add_argument("--format", choices=["png", "svg"], default="png")
args = parser.parse_args()

names = ta.get_all_activations()
generated = []
skipped = []

for name in names:
    cls = ta._ACTIVATIONS[name]["class"]
    try:
        plot_activation(cls, params={}, save_dir=args.output_dir, fmt=args.format)
        generated.append(name)
    except Exception as e:
        print(f"Warning: skipping {name}: {e}")
        skipped.append(name)

print(f"\n{len(generated)} generated, {len(skipped)} skipped")
if skipped:
    print("Skipped:", ", ".join(skipped))
