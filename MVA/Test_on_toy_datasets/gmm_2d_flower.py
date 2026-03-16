import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
import yaml

from MVA.Test_on_toy_datasets.gmm_2d_common import (
    GMM2D,
    load_config,
    load_model_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Flower on the 2D GMM toy experiments using a saved flow-matching checkpoint."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="MVA/Test_on_toy_datasets/configs/gmm_2d_default.yaml",
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def make_grid(xlim, ylim, grid_size):
    xs = np.linspace(xlim[0], xlim[1], grid_size)
    ys = np.linspace(ylim[0], ylim[1], grid_size)
    xx, yy = np.meshgrid(xs, ys)
    points = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    return xx, yy, points


def style_axis(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#999999")
        spine.set_linewidth(0.8)


def draw_measurement_line(ax, h, y, xlim, ylim):
    h = np.asarray(h, dtype=float).reshape(-1)
    h0, h1 = h

    if abs(h1) > 1e-8:
        xs = np.linspace(xlim[0], xlim[1], 300)
        ys = (y - h0 * xs) / h1
        ax.plot(xs, ys, linestyle="--", color="black", linewidth=1.2)
    elif abs(h0) > 1e-8:
        x_const = y / h0
        ax.plot([x_const, x_const], [ylim[0], ylim[1]], linestyle="--", color="black", linewidth=1.2)


def add_shared_legend(fig):
    handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=5, markerfacecolor="#7f7f7f", markeredgewidth=0, alpha=0.8),
        Line2D([0], [0], marker="o", linestyle="None", markersize=5, markerfacecolor="#4ea5ff", markeredgewidth=0, alpha=0.8),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.2),
    ]
    labels = [
        "Prior GMM samples",
        "Flower samples",
        "Constraint h^T x = y",
    ]
    fig.legend(
        handles,
        labels,
        loc="center left",
        ncol=1,
        frameon=False,
        bbox_to_anchor=(0.86, 0.28),
        fontsize=10,
    )


def model_velocity(model, particles, t_scalar, device):
    with torch.no_grad():
        x = torch.tensor(particles, dtype=torch.float32, device=device)
        t = torch.full((particles.shape[0],), float(t_scalar), dtype=torch.float32, device=device)
        velocity = model(x, t).cpu().numpy()
    return velocity


def destination_refinement(particles, h, y, sigma_n, lam, gamma, rng):
    h = np.asarray(h, dtype=float).reshape(1, -1)
    dim = particles.shape[1]
    system = (h.T @ h) / (sigma_n ** 2) + np.eye(dim) / lam
    rhs = (h.T * y).reshape(1, dim) / (sigma_n ** 2) + particles / lam
    mu = np.linalg.solve(system, rhs.T).T

    if gamma == 0:
        return mu

    cov = np.linalg.inv(system)
    noise = rng.multivariate_normal(mean=np.zeros(dim), cov=cov, size=particles.shape[0])
    return mu + noise


def flower_one_step(model, particles, t, delta, h, y, sigma_n, gamma, device, rng):
    velocity = model_velocity(model, particles, t, device)
    step1 = particles + (1.0 - t) * velocity

    sigma_r = (1.0 - t) / np.sqrt(t ** 2 + (1.0 - t) ** 2)
    lam = sigma_r ** 2
    step2 = destination_refinement(step1, h, y, sigma_n, lam, gamma, rng)

    z0 = rng.standard_normal(particles.shape)
    step3 = (t + delta) * step2 + (1.0 - t - delta) * z0
    return step1, step2, step3


def run_flower(model, num_particles, num_steps, h, y, sigma_n, gamma, device, rng, selected_iterations):
    delta = 1.0 / num_steps
    particles = rng.standard_normal((num_particles, 2))
    history = []
    trajectory = [particles[0].copy()]

    for iteration in range(num_steps):
        t = iteration * delta
        step1, step2, step3 = flower_one_step(
            model=model,
            particles=particles,
            t=t,
            delta=delta,
            h=h,
            y=y,
            sigma_n=sigma_n,
            gamma=gamma,
            device=device,
            rng=rng,
        )
        if iteration in selected_iterations:
            history.append(
                {
                    "iteration": iteration,
                    "t": t,
                    "step1": step1.copy(),
                    "step2": step2.copy(),
                    "step3": step3.copy(),
                }
            )
        particles = step3
        trajectory.append(particles[0].copy())

    return particles, history, np.asarray(trajectory)


def save_path_figure(
    history,
    prior_samples,
    posterior_samples,
    xlim,
    ylim,
    h,
    y,
    scenario_title,
    gamma,
    output_path,
):
    fig, axes = plt.subplots(
        3,
        len(history),
        figsize=(2.15 * len(history) + 4.0, 6.8),
    )
    fig.subplots_adjust(left=0.06, right=0.82, top=0.88, bottom=0.12, wspace=0.08, hspace=0.08)
    if len(history) == 1:
        axes = axes[:, None]

    row_labels = [
        "Step 1: x1(x_t)",
        f"gamma = {gamma}\nStep 2: x(x_t, y)",
        "Step 3: x_{t+Delta}",
    ]
    fig.suptitle(f"{scenario_title} | Flower temporal evolution | gamma = {gamma}", fontsize=15)

    for col, item in enumerate(history):
        t_label = f"t = {item['t']:.3f}"
        axes[0, col].set_title(t_label, fontsize=11, pad=4)
        clouds = [item["step1"], item["step2"], item["step3"]]
        for row, cloud in enumerate(clouds):
            ax = axes[row, col]
            ax.scatter(
                prior_samples[:, 0],
                prior_samples[:, 1],
                s=2,
                alpha=0.18,
                color="#7f7f7f",
                linewidths=0,
                zorder=1,
            )
            ax.scatter(
                cloud[:, 0],
                cloud[:, 1],
                s=2,
                alpha=0.35,
                color="#4ea5ff",
                linewidths=0,
                zorder=2,
            )
            draw_measurement_line(ax, h, y, xlim, ylim)
            style_axis(ax, xlim, ylim)
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=11)

    inset_ax = fig.add_axes([0.84, 0.47, 0.14, 0.22])
    inset_ax.scatter(
        prior_samples[:, 0],
        prior_samples[:, 1],
        s=2,
        alpha=0.14,
        color="#7f7f7f",
        linewidths=0,
        zorder=1,
    )
    inset_ax.scatter(
        posterior_samples[:, 0],
        posterior_samples[:, 1],
        s=4,
        alpha=0.30,
        color="#ff6b6b",
        linewidths=0,
        zorder=2,
    )
    draw_measurement_line(inset_ax, h, y, xlim, ylim)
    style_axis(inset_ax, xlim, ylim)
    inset_ax.set_title("True Posterior", fontsize=10, pad=2)

    add_shared_legend(fig)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_comparison_figure(
    posterior_samples,
    prior_samples,
    gamma0_samples,
    gamma1_samples,
    posterior_density,
    xx,
    yy,
    xlim,
    ylim,
    h,
    y,
    scenario_title,
    output_path,
):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.2))
    fig.subplots_adjust(left=0.05, right=0.82, top=0.88, bottom=0.12, wspace=0.12)
    fig.suptitle(f"{scenario_title} | Posterior comparison", fontsize=16)
    panels = [
        ("True posterior samples", posterior_samples, "#2a9d8f"),
        ("Flower gamma = 0", gamma0_samples, "#e76f51"),
        ("Flower gamma = 1", gamma1_samples, "#f4a261"),
    ]
    for ax, (title, samples, color) in zip(axes, panels):
        ax.contourf(xx, yy, posterior_density, levels=30, cmap="Blues", alpha=0.92)
        ax.contour(xx, yy, posterior_density, levels=8, colors="white", linewidths=0.45, alpha=0.65)
        ax.set_title(title)
        ax.scatter(prior_samples[:, 0], prior_samples[:, 1], s=2, alpha=0.12, color="#7f7f7f", linewidths=0)
        ax.scatter(samples[:, 0], samples[:, 1], s=7, alpha=0.22, color=color, linewidths=0)
        draw_measurement_line(ax, h, y, xlim, ylim)
        style_axis(ax, xlim, ylim)

    add_shared_legend(fig)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_summary(config, checkpoint_metadata, summaries, output_dir):
    summary = {
        "config": config,
        "checkpoint_path": config["training"]["checkpoint_path"],
        "checkpoint_hidden_dim": checkpoint_metadata["hidden_dim"],
        "scenarios": summaries,
    }
    with open(output_dir / "run_summary.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)


def save_single_trajectory_figure(
    trajectory,
    prior_samples,
    posterior_samples,
    xlim,
    ylim,
    h,
    y,
    scenario_title,
    gamma,
    output_path,
):
    time_grid = np.linspace(0.0, 1.0, len(trajectory))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.12, wspace=0.20)
    ax = axes[0]
    ax_t = axes[1]

    ax.scatter(
        prior_samples[:, 0],
        prior_samples[:, 1],
        s=2,
        alpha=0.10,
        color="#7f7f7f",
        linewidths=0,
        zorder=1,
        label="Prior GMM samples",
    )
    ax.scatter(
        posterior_samples[:, 0],
        posterior_samples[:, 1],
        s=4,
        alpha=0.18,
        color="#ff6b6b",
        linewidths=0,
        zorder=2,
        label="True posterior",
    )
    ax.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        color="#1d4ed8",
        linewidth=2.2,
        alpha=0.9,
        zorder=4,
        label="Trajectory of one x_t",
    )
    ax.scatter(
        [trajectory[0, 0]],
        [trajectory[0, 1]],
        s=50,
        color="#0f172a",
        marker="o",
        zorder=6,
        label="Start t=0",
    )
    ax.scatter(
        [trajectory[-1, 0]],
        [trajectory[-1, 1]],
        s=60,
        color="#2563eb",
        marker="X",
        zorder=7,
        label="End t=1",
    )
    draw_measurement_line(ax, h, y, xlim, ylim)
    style_axis(ax, xlim, ylim)
    ax.set_title("Trajectory in the 2D plane", fontsize=12)
    ax.legend(loc="upper right", frameon=False, fontsize=9)

    ax_t.plot(time_grid, trajectory[:, 0], color="#1d4ed8", linewidth=2.0, label="x_1(t)")
    ax_t.plot(time_grid, trajectory[:, 1], color="#ef4444", linewidth=2.0, label="x_2(t)")
    ax_t.scatter([time_grid[0]], [trajectory[0, 0]], color="#1d4ed8", s=30, zorder=3)
    ax_t.scatter([time_grid[-1]], [trajectory[-1, 0]], color="#1d4ed8", s=40, marker="X", zorder=3)
    ax_t.scatter([time_grid[0]], [trajectory[0, 1]], color="#ef4444", s=30, zorder=3)
    ax_t.scatter([time_grid[-1]], [trajectory[-1, 1]], color="#ef4444", s=40, marker="X", zorder=3)
    ax_t.set_title("Coordinates over time", fontsize=12)
    ax_t.set_xlabel("t")
    ax_t.set_ylabel("coordinate value")
    ax_t.grid(alpha=0.25)
    ax_t.legend(frameon=False, fontsize=9)

    fig.suptitle(f"{scenario_title} | One sample trajectory | gamma = {gamma}", fontsize=14)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    config = load_config(args.config)

    experiment_cfg = config["experiment"]
    gmm_cfg = config["gmm"]
    sampling_cfg = config["sampling"]
    plot_cfg = config["plot"]
    scenarios_cfg = config["scenarios"]
    checkpoint_path = Path(config["training"]["checkpoint_path"])

    output_dir = Path(experiment_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. "
            "Run MVA/Test_on_toy_datasets/train_gmm_2d_model.py first."
        )

    rng = np.random.default_rng(experiment_cfg["seed"])
    torch.manual_seed(experiment_cfg["seed"])
    device = torch.device(experiment_cfg["device"])

    prior = GMM2D(
        gmm_cfg["weights"],
        gmm_cfg["means"],
        gmm_cfg["covariances"],
    )
    model, checkpoint_metadata = load_model_checkpoint(checkpoint_path, device)
    prior_reference_samples, _ = prior.sample(sampling_cfg.get("num_prior_plot_samples", 12000), rng)

    xx, yy, grid_points = make_grid(plot_cfg["xlim"], plot_cfg["ylim"], plot_cfg["grid_size"])
    scenario_summaries = []

    for scenario_cfg in scenarios_cfg:
        name = scenario_cfg["name"]
        h = np.asarray(scenario_cfg["h"], dtype=float)
        y_obs = float(scenario_cfg["y"])
        sigma_n = float(scenario_cfg["noise_std"])

        posterior = prior.posterior_given_linear_observation(h, y_obs, sigma_n)
        posterior_samples, _ = posterior.sample(sampling_cfg["num_posterior_samples"], rng)
        posterior_density = np.exp(posterior.logpdf(grid_points)).reshape(xx.shape)

        gamma0_samples, gamma0_history, gamma0_trajectory = run_flower(
            model=model,
            num_particles=sampling_cfg["num_particles"],
            num_steps=sampling_cfg["num_steps"],
            h=h,
            y=y_obs,
            sigma_n=sigma_n,
            gamma=0,
            device=device,
            rng=rng,
            selected_iterations=set(sampling_cfg["selected_iterations"]),
        )
        gamma1_samples, gamma1_history, gamma1_trajectory = run_flower(
            model=model,
            num_particles=sampling_cfg["num_particles"],
            num_steps=sampling_cfg["num_steps"],
            h=h,
            y=y_obs,
            sigma_n=sigma_n,
            gamma=1,
            device=device,
            rng=rng,
            selected_iterations=set(sampling_cfg["selected_iterations"]),
        )

        save_comparison_figure(
            posterior_samples=posterior_samples,
            prior_samples=prior_reference_samples,
            gamma0_samples=gamma0_samples,
            gamma1_samples=gamma1_samples,
            posterior_density=posterior_density,
            xx=xx,
            yy=yy,
            xlim=plot_cfg["xlim"],
            ylim=plot_cfg["ylim"],
            h=h,
            y=y_obs,
            scenario_title=f"{name} | h=[{h[0]:.2f}, {h[1]:.2f}] | y={y_obs:.2f} | sigma_n={sigma_n:.2f}",
            output_path=output_dir / f"{name}_posterior_comparison.png",
        )
        save_path_figure(
            history=gamma0_history,
            prior_samples=prior_reference_samples,
            posterior_samples=posterior_samples,
            xlim=plot_cfg["xlim"],
            ylim=plot_cfg["ylim"],
            h=h,
            y=y_obs,
            scenario_title=f"{name} | h=[{h[0]:.2f}, {h[1]:.2f}] | y={y_obs:.2f} | sigma_n={sigma_n:.2f}",
            gamma=0,
            output_path=output_dir / f"{name}_flower_paths_gamma_0.png",
        )
        save_single_trajectory_figure(
            trajectory=gamma0_trajectory,
            prior_samples=prior_reference_samples,
            posterior_samples=posterior_samples,
            xlim=plot_cfg["xlim"],
            ylim=plot_cfg["ylim"],
            h=h,
            y=y_obs,
            scenario_title=f"{name} | h=[{h[0]:.2f}, {h[1]:.2f}] | y={y_obs:.2f} | sigma_n={sigma_n:.2f}",
            gamma=0,
            output_path=output_dir / f"{name}_single_trajectory_gamma_0.png",
        )
        save_path_figure(
            history=gamma1_history,
            prior_samples=prior_reference_samples,
            posterior_samples=posterior_samples,
            xlim=plot_cfg["xlim"],
            ylim=plot_cfg["ylim"],
            h=h,
            y=y_obs,
            scenario_title=f"{name} | h=[{h[0]:.2f}, {h[1]:.2f}] | y={y_obs:.2f} | sigma_n={sigma_n:.2f}",
            gamma=1,
            output_path=output_dir / f"{name}_flower_paths_gamma_1.png",
        )
        save_single_trajectory_figure(
            trajectory=gamma1_trajectory,
            prior_samples=prior_reference_samples,
            posterior_samples=posterior_samples,
            xlim=plot_cfg["xlim"],
            ylim=plot_cfg["ylim"],
            h=h,
            y=y_obs,
            scenario_title=f"{name} | h=[{h[0]:.2f}, {h[1]:.2f}] | y={y_obs:.2f} | sigma_n={sigma_n:.2f}",
            gamma=1,
            output_path=output_dir / f"{name}_single_trajectory_gamma_1.png",
        )

        scenario_summaries.append(
            {
                "name": name,
                "h": [float(v) for v in h],
                "y": y_obs,
                "sigma_n": sigma_n,
                "posterior_weights": [float(v) for v in posterior.weights],
            }
        )

    save_summary(config, checkpoint_metadata, scenario_summaries, output_dir)


if __name__ == "__main__":
    main()
