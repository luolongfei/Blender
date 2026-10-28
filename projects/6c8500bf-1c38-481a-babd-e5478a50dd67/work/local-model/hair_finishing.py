"""Shallow rear gather locks and restrained nape wisps.

Adds ten new meshes; does not replace, delete, or edit existing hairstyle objects.
The centre part fans toward the low ties instead of ending as a horizontal bob.
"""
import importlib.util
import math
from pathlib import Path


def _helpers():
    path = Path(__file__).with_name("hair.py")
    spec = importlib.util.spec_from_file_location("charakuru_hair_finishing_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _back(x, z, lift=0.0055):
    """A point just outside the scalp ellipse, on the visible rear surface."""
    q = max(0.0, 1 - (x / 0.121) ** 2 - ((z - 1.450) / 0.145) ** 2)
    return (x, 0.004 + 0.108 * math.sqrt(q) + lift, z)


def build_back_hair(collection, material):
    hair = _helpers()
    objects = []
    for sign, side in ((-1, "R"), (1, "L")):
        paths = [
            [
                _back(sign * 0.003, 1.563, 0.003),
                _back(sign * 0.029, 1.534),
                _back(sign * 0.066, 1.482),
                _back(sign * 0.088, 1.435),
                (sign * 0.101, 0.062, 1.402),
                (sign * 0.108, 0.046, 1.383),
            ],
            [
                _back(sign * 0.002, 1.531, 0.003),
                _back(sign * 0.016, 1.493),
                _back(sign * 0.046, 1.448),
                _back(sign * 0.074, 1.416),
                (sign * 0.095, 0.067, 1.394),
                (sign * 0.106, 0.047, 1.381),
            ],
            [
                _back(sign * 0.0015, 1.491, 0.003),
                _back(sign * 0.010, 1.454),
                _back(sign * 0.033, 1.418),
                _back(sign * 0.060, 1.394),
                (sign * 0.083, 0.078, 1.381),
                (sign * 0.103, 0.047, 1.380),
            ],
        ]
        for i, controls in enumerate(paths):
            objects += hair._lock(
                "Hair_BackGather_%s_%02d" % (side, i), controls,
                width=(0.026, 0.030, 0.026)[i],
                thickness=0.00145,
                normal=hair._scalp_normal,
                material=material,
                collection=collection,
                side=side,
                region="scalp",
                strands=0,
                root_ratio=0.015,
                peak_at=0.48,
                root_embed=0.005,
                tip_length=0.30,
                steps=26,
                section=10,
            )

        # Four short, fine wisps break the mechanical lower edge without a fringe
        # of spikes. They follow the rear surface and gently lean into the gathers.
        nape = [
            [
                _back(sign * 0.006, 1.419, 0.003),
                _back(sign * 0.011, 1.397, 0.005),
                _back(sign * 0.020, 1.378, 0.004),
                (sign * 0.028, 0.086, 1.360),
            ],
            [
                _back(sign * 0.030, 1.411, 0.003),
                _back(sign * 0.038, 1.392, 0.005),
                _back(sign * 0.046, 1.379, 0.004),
                (sign * 0.054, 0.080, 1.368),
            ],
        ]
        for i, controls in enumerate(nape):
            objects += hair._lock(
                "Hair_NapeWisp_%s_%02d" % (side, i), controls,
                width=0.0068,
                thickness=0.0008,
                normal=(0, 1, 0),
                material=material,
                collection=collection,
                side=side,
                region="scalp",
                strands=0,
                root_ratio=0.02,
                peak_at=0.40,
                root_embed=0.004,
                tip_length=0.42,
                steps=22,
                section=8,
            )
    for obj in objects:
        obj["part"] = "hair"
        obj["hair_region"] = "scalp"
        obj["rig_attachment"] = "head"
        obj["finish_note"] = "Rear centre part gathered toward low ponytail ties"
    return objects
