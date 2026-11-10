"""Run each supplied dataset/seed checkpoint in its own process and output folder."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from runtime import PROJECT_ROOT, project_path

DATASETS = {"harvard": ("test_medical.py", "configs/harvard.yaml", "harvard"),
            "msrs": ("test_msrs.py", "configs/msrs.yaml", "msrs"),
            "llvip": ("test_llvip.py", "configs/llvip_lessnew_test.yaml", "llvip_lessnew")}
SEEDS = (42, 123, 3407)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", default="degraded", choices=("clean", "degraded", "both"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-root", default="output/all_checkpoints")
    parser.add_argument("--data-root", help="Parent of harvard, msrs, and llvip_lessnew directories")
    parser.add_argument("--pretrained-root")
    parser.add_argument("--captions-root", help="Layout: DATASET/seedSEED/test_CONDITION.json")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--save-stages", action="store_true")
    parser.add_argument("--skip-precompute", action="store_true")
    parser.add_argument("--list", action="store_true", help="Print commands without executing inference")
    args = parser.parse_args()
    conditions = ("clean", "degraded") if args.condition == "both" else (args.condition,)
    results = []
    out_root = project_path(args.out_root)
    for dataset, (script, config, data_folder) in DATASETS.items():
        for seed in SEEDS:
            checkpoint = PROJECT_ROOT / "checkpoints" / dataset / f"seed{seed}" / "model_6000.pkl"
            if not checkpoint.is_file():
                parser.exit(1, f"Missing supplied checkpoint: {checkpoint}\n")
            for condition in conditions:
                command = [sys.executable, str(PROJECT_ROOT / script), "--config", str(PROJECT_ROOT / config),
                           "--checkpoint", str(checkpoint), "--condition", condition,
                           "--device", args.device, "--out-root", str(out_root / dataset / f"seed{seed}")]
                if args.data_root:
                    command += ["--data-dir", str(Path(args.data_root).resolve() / data_folder)]
                if args.pretrained_root:
                    command += ["--pretrained-root", str(Path(args.pretrained_root).resolve())]
                if args.captions_root:
                    command += ["--captions", str(Path(args.captions_root).resolve() / dataset /
                                                 f"seed{seed}" / f"test_{condition}.json")]
                if args.max_samples is not None:
                    command += ["--max-samples", str(args.max_samples)]
                if args.save_stages:
                    command.append("--save-stages")
                if args.skip_precompute:
                    command.append("--skip-precompute")
                print(subprocess.list2cmdline(command), flush=True)
                if args.list:
                    continue
                result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
                results.append({"dataset": dataset, "seed": seed, "condition": condition,
                                "returncode": result.returncode, "command": command})
                out_root.mkdir(parents=True, exist_ok=True)
                (out_root / "batch_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
                if result.returncode:
                    parser.exit(result.returncode, f"Stopped after {dataset}/seed{seed}/{condition} failed.\n")


if __name__ == "__main__":
    main()
