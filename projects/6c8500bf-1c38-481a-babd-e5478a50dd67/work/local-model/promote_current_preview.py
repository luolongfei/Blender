"""Point legacy fixed export paths at the verified current preview.

Keep the previous bytes under one checkpoint before any replacement. The
expression-review folder remains the canonical preview output directory.
"""
from pathlib import Path
import hashlib
import json
import shutil

P = Path(__file__).resolve().parents[2]
E = P / "exports"
CURRENT = E / "expression-review"
CHECKPOINT = P / "work/checkpoints/comprehensive-refinement-start/legacy-export-aliases"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    report = json.loads((CURRENT / "avatar-export-report.json").read_text())
    source_sha = sha(P / "work/character.blend")
    if report["sourceBlendSha256"] != source_sha:
        raise ValueError("Current preview was not exported from the saved master.")
    for name, item in report["files"].items():
        if sha(CURRENT / name) != item["sha256"]:
            raise ValueError("Preview file changed: " + name)
    names = [
        "avatar-preview.vrm", "avatar-preview.glb", "avatar-runtime.blend",
        "avatar-export-report.json", "avatar-spring-plan.json",
        "avatar-physics-report.json",
    ]
    CHECKPOINT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in names:
        original, replacement = E / name, CURRENT / name
        previous = sha(original) if original.is_file() else None
        if original.is_file() and previous != sha(replacement):
            retained = CHECKPOINT / name
            if retained.exists() and sha(retained) != previous:
                raise ValueError("Earlier legacy alias checkpoint differs: " + name)
            if not retained.exists():
                shutil.copy2(original, retained)
        stage = E / (name + ".current")
        shutil.copy2(replacement, stage)
        if sha(stage) != sha(replacement):
            raise ValueError("Copy failed checksum: " + name)
        stage.replace(original)
        rows.append({
            "file": name, "previousSha256": previous,
            "currentSha256": sha(original),
            "canonicalPath": str(replacement.relative_to(P)),
        })
    result = {
        "sourceBlendSha256": source_sha,
        "vrmSha256": sha(CURRENT / "avatar-preview.vrm"),
        "aliases": rows,
        "olderRootDiagnosticReportsAreHistorical": True,
        "currentDiagnostics": "exports/expression-review/",
    }
    (E / "current-preview-aliases.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    run()
