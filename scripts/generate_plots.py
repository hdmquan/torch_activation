import torch
import torch_activation
from torch_activation.utils import plot_activation

NEEDS_INPUT_SHAPE = {
    "GPSoftmax",
    "GLSoftmax",
    "ARBF",
    "PGELU",
    "PFTS",
    "PFPM",
    "PSIGRAMP",
    "RSIGN",
    "MAF",
    "UAF",
    "GReLU",
    "GLN",
}

DOUBLES_FIRST_DIM = {"CReLU", "NCReLU"}
DOUBLES_LAST_DIM = {"PairedReLU"}
SKIP = {"BaseActivation"}

names = [n for n in torch_activation._ACTIVATIONS.keys() if n not in SKIP]
total = len(names)
generated = 0
failed = 0

for i, name in enumerate(names, 1):
    cls = torch_activation._ACTIVATIONS[name]["class"]
    print(f"[{i}/{total}] {cls.__name__}")
    try:
        if name in NEEDS_INPUT_SHAPE:
            m = cls(input_shape=4)
        else:
            m = cls()
        plot_activation(m, params={})
        generated += 1
    except Exception as e:
        print(f"  WARNING: {name} failed: {e}")
        failed += 1

print(f"\nGenerated: {generated}, Failed: {failed}")
