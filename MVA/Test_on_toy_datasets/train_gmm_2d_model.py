import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from MVA.Test_on_toy_datasets.gmm_2d_common import (
    GMM2D,
    load_config,
    save_model_checkpoint,
    save_training_curve,
    train_velocity_model,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and save the flow-matching network for the 2D GMM toy experiments."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="MVA/Test_on_toy_datasets/configs/gmm_2d_default.yaml",
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def save_training_summary(config, loss_history, output_dir):
    summary = {
        "config": config,
        "training_loss_history": [{"step": step, "loss": loss} for step, loss in loss_history],
        "checkpoint_path": config["training"]["checkpoint_path"],
    }
    with open(output_dir / "training_summary.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)


def main():
    args = parse_args()
    config = load_config(args.config)

    experiment_cfg = config["experiment"]
    training_cfg = config["training"]
    gmm_cfg = config["gmm"]

    output_dir = Path(training_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(experiment_cfg["seed"])
    torch.manual_seed(experiment_cfg["seed"])
    device = torch.device(experiment_cfg["device"])

    prior = GMM2D(
        gmm_cfg["weights"],
        gmm_cfg["means"],
        gmm_cfg["covariances"],
    )

    model, loss_history = train_velocity_model(prior, training_cfg, device, rng)
    save_model_checkpoint(model, config, training_cfg["checkpoint_path"])
    save_training_curve(loss_history, output_dir / "training_curve.png")
    save_training_summary(config, loss_history, output_dir)


if __name__ == "__main__":
    main()
