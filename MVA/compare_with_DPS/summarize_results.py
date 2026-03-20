import argparse
import json
from pathlib import Path


def _read_last_metric(path: Path, key: str) -> float:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 2:
        raise RuntimeError(f"No metric values found in {path}")

    header = lines[0].split()
    if key not in header:
        raise RuntimeError(f"Column '{key}' not found in {path}")

    idx = header.index(key)
    last_values = lines[-1].split()
    return float(last_values[idx])


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Flower and DPS comparison metrics")
    parser.add_argument("--flower_psnr", type=str, required=True)
    parser.add_argument("--flower_ssim", type=str, required=True)
    parser.add_argument("--flower_lpips", type=str, required=True)
    parser.add_argument("--dps_metrics", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    args = parser.parse_args()

    flower_psnr = _read_last_metric(Path(args.flower_psnr), "psnr_rec")
    flower_ssim = _read_last_metric(Path(args.flower_ssim), "ssim_rec")
    flower_lpips = _read_last_metric(Path(args.flower_lpips), "lpips_rec")

    dps_path = Path(args.dps_metrics)
    if not dps_path.exists():
        raise FileNotFoundError(f"Missing DPS metrics file: {dps_path}")
    dps_metrics = json.loads(dps_path.read_text(encoding="utf-8"))

    summary = {
        "flower": {
            "psnr_rec": flower_psnr,
            "ssim_rec": flower_ssim,
            "lpips_rec": flower_lpips,
        },
        "dps": dps_metrics,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
