from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def logsumexp(values, axis=-1, keepdims=False):
    vmax = np.max(values, axis=axis, keepdims=True)
    stable = values - vmax
    summed = np.log(np.sum(np.exp(stable), axis=axis, keepdims=True)) + vmax
    if keepdims:
        return summed
    return np.squeeze(summed, axis=axis)


def gaussian_logpdf(x, mean, cov):
    dim = mean.shape[0]
    diff = x - mean
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise ValueError("Covariance matrix must be positive definite.")
    inv_cov = np.linalg.inv(cov)
    mahal = np.einsum("...i,ij,...j->...", diff, inv_cov, diff)
    return -0.5 * (dim * np.log(2.0 * np.pi) + logdet + mahal)


class GMM2D:
    def __init__(self, weights, means, covariances):
        self.weights = np.asarray(weights, dtype=float)
        self.weights = self.weights / self.weights.sum()
        self.log_weights = np.log(self.weights)
        self.means = np.asarray(means, dtype=float)
        self.covariances = np.asarray(covariances, dtype=float)
        self.num_components = self.weights.shape[0]
        self.dim = self.means.shape[1]

    def sample_components(self, num_samples, rng):
        return rng.choice(self.num_components, size=num_samples, p=self.weights)

    def sample(self, num_samples, rng):
        component_ids = self.sample_components(num_samples, rng)
        samples = np.zeros((num_samples, self.dim), dtype=float)
        for k in range(self.num_components):
            mask = component_ids == k
            count = int(mask.sum())
            if count == 0:
                continue
            samples[mask] = rng.multivariate_normal(
                mean=self.means[k],
                cov=self.covariances[k],
                size=count,
            )
        return samples, component_ids

    def logpdf(self, x):
        logs = []
        for k in range(self.num_components):
            logs.append(self.log_weights[k] + gaussian_logpdf(x, self.means[k], self.covariances[k]))
        return logsumexp(np.stack(logs, axis=-1), axis=-1)

    def posterior_given_linear_observation(self, h, y, noise_std):
        h = np.asarray(h, dtype=float).reshape(1, -1)
        y = np.asarray([y], dtype=float)
        noise_cov = np.asarray([[noise_std ** 2]], dtype=float)
        noise_precision = np.linalg.inv(noise_cov)

        posterior_weights = []
        posterior_means = []
        posterior_covariances = []

        for k in range(self.num_components):
            prior_mean = self.means[k]
            prior_cov = self.covariances[k]

            predictive_cov = h @ prior_cov @ h.T + noise_cov
            predictive_mean = h @ prior_mean
            log_weight = self.log_weights[k] + gaussian_logpdf(
                y[None, :], predictive_mean, predictive_cov
            )[0]

            posterior_precision = np.linalg.inv(prior_cov) + h.T @ noise_precision @ h
            posterior_cov = np.linalg.inv(posterior_precision)
            innovation = y - predictive_mean
            posterior_mean = prior_mean + prior_cov @ h.T @ np.linalg.solve(predictive_cov, innovation)

            posterior_weights.append(log_weight)
            posterior_means.append(posterior_mean)
            posterior_covariances.append(posterior_cov)

        posterior_weights = np.asarray(posterior_weights)
        posterior_weights = np.exp(posterior_weights - logsumexp(posterior_weights))

        return GMM2D(
            posterior_weights,
            np.asarray(posterior_means),
            np.asarray(posterior_covariances),
        )


class VelocityMLP(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x, t):
        if t.ndim == 1:
            t = t[:, None]
        return self.net(torch.cat([x, t], dim=1))


def sample_flow_matching_batch(prior, batch_size, rng, device):
    x1_np, _ = prior.sample(batch_size, rng)
    z_np = rng.standard_normal((batch_size, 2))
    t_np = rng.random(batch_size)
    xt_np = t_np[:, None] * x1_np + (1.0 - t_np[:, None]) * z_np
    target_np = x1_np - z_np

    x_t = torch.tensor(xt_np, dtype=torch.float32, device=device)
    t = torch.tensor(t_np, dtype=torch.float32, device=device)
    target = torch.tensor(target_np, dtype=torch.float32, device=device)
    return x_t, t, target


def train_velocity_model(prior, training_cfg, device, rng):
    model = VelocityMLP(training_cfg["hidden_dim"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_cfg["lr"])
    loss_history = []

    for step in range(training_cfg["train_steps"]):
        x_t, t, target = sample_flow_matching_batch(prior, training_cfg["batch_size"], rng, device)
        prediction = model(x_t, t)
        loss = torch.mean((prediction - target) ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % training_cfg["log_every"] == 0 or step == training_cfg["train_steps"] - 1:
            loss_history.append((step, float(loss.detach().cpu().item())))

    return model, loss_history


def save_training_curve(loss_history, output_path):
    steps = [step for step, _ in loss_history]
    losses = [loss for _, loss in loss_history]
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.plot(steps, losses, color="#1d3557", linewidth=2)
    ax.set_title("Flow-matching training loss")
    ax.set_xlabel("Training step")
    ax.set_ylabel("MSE")
    ax.grid(alpha=0.25)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_model_checkpoint(model, config, checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "hidden_dim": config["training"]["hidden_dim"],
            "config": config,
        },
        checkpoint_path,
    )


def load_model_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = VelocityMLP(checkpoint["hidden_dim"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint
