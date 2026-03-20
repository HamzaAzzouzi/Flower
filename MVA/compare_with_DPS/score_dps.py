import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _load_rgb(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr


def _compute_metrics(label: np.ndarray, recon: np.ndarray) -> tuple[float, float]:
    psnr = peak_signal_noise_ratio(label, recon, data_range=1.0)
    ssim = structural_similarity(label, recon, channel_axis=2, data_range=1.0)
    return float(psnr), float(ssim)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute DPS PSNR/SSIM from saved recon and label images")
    parser.add_argument("--label_dir", type=str, required=True)
    parser.add_argument("--recon_dir", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    args = parser.parse_args()

    label_dir = Path(args.label_dir)
    recon_dir = Path(args.recon_dir)

    if not label_dir.exists() or not recon_dir.exists():
        raise FileNotFoundError("label_dir or recon_dir does not exist")

    label_paths = sorted([p for p in label_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    recon_paths = sorted([p for p in recon_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])

    if len(label_paths) == 0 or len(recon_paths) == 0:
        raise RuntimeError("No images found in label_dir or recon_dir")

    common_names = sorted(set(p.name for p in label_paths) & set(p.name for p in recon_paths))
    if len(common_names) == 0:
        raise RuntimeError("No matching filenames between label_dir and recon_dir")

    psnr_values = []
    ssim_values = []

    for name in common_names:
        label = _load_rgb(label_dir / name)
        recon = _load_rgb(recon_dir / name)

        if label.shape != recon.shape:
            raise ValueError(f"Shape mismatch for {name}: {label.shape} vs {recon.shape}")

        psnr, ssim = _compute_metrics(label, recon)
        psnr_values.append(psnr)
        ssim_values.append(ssim)

    result = {
        "num_images": len(common_names),
        "psnr_rec": float(np.mean(psnr_values)),
        "ssim_rec": float(np.mean(ssim_values)),
        "lpips_rec": None,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
