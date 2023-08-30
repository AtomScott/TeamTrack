import wandb
import pytorch_lightning as pl
import torch
from torchmetrics.functional import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
from teamtrack.base_module import BaseMotionModule
from teamtrack.metrics import trajectory_nll_loss, rmse_loss
from einops import rearrange


class MDNMotionModule(BaseMotionModule):
    def __init__(self, model, learning_rate=1e-3, roll_out_steps=10, single_agent=False, sport="soccer", original_fps=25, inference_fps=25):
        super().__init__(model, learning_rate, roll_out_steps, single_agent, sport, original_fps, inference_fps)

    def compute_and_log_losses(self, x, y, outputs, stage, suffix=""):
        y_pred, pi, mu, sigma, rhos = outputs
        if not self.single_agent:
            x = rearrange(x, "B L N M -> (B N) L M")
            y = rearrange(y, "B L N M -> (B N) L M")
            # y_pred = rearrange(y_pred, "B N M -> (B N) M")
            # pi = rearrange(pi, "B N M -> (B N) M")
            # mu = rearrange(mu, "B N M -> (B N) M")
            # sigma = rearrange(sigma, "B N M -> (B N) M")
            # rhos = rearrange(rhos, "B N M -> (B N) M")

        mu_x, mu_y = torch.split(mu, [self.model.n_mixtures, self.model.n_mixtures], 2)
        sigma_x, sigma_y = torch.split(sigma, [self.model.n_mixtures, self.model.n_mixtures], 2)

        y_disp = torch.concatenate([x[:, -1:], y], axis=1).diff(axis=1)

        nll = trajectory_nll_loss(y_disp, pi, mu_x, mu_y, sigma_x, sigma_y, rhos)

        y = self.upsample(y)
        y_pred = self.upsample(y_pred)
        rmse = rmse_loss(y_gt=y, y_pred=y_pred)

        self.log(f"{stage}_nll_loss{suffix}", nll, prog_bar=True)
        self.log(f"{stage}_rmse_loss{suffix}", rmse, prog_bar=True)

        self.log(f"mu_x", mu_x.mean())
        self.log(f"mu_y", mu_y.mean())
        return nll, rmse

    def training_step(self, batch, batch_idx):
        x, y = batch

        x = self.downsample(x)
        y = self.downsample(y)

        # Roll out predictions using teacher forcing
        outputs = self.model.roll_out(x, n_steps=y.shape[1], y_gt=y, return_all_params=True)
        nll, _ = self.compute_and_log_losses(x, y, outputs, "train")

        return nll

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = self.downsample(x)
        y = self.downsample(y)

        # Roll out predictions with teacher forcing
        outputs = self.model.roll_out(x, n_steps=y.shape[1], y_gt=y, return_all_params=True)
        self.compute_and_log_losses(x, y, outputs, "val")

        # Roll out predictions without teacher forcing
        outputs_notf = self.model.roll_out(x, n_steps=y.shape[1], y_gt=None, return_all_params=True)
        self.compute_and_log_losses(x, y, outputs_notf, "val", "_notf")

        if batch_idx == 0:
            if not self.single_agent:
                x = rearrange(x, "B L N M -> (B N) L M")
                y = rearrange(y, "B L N M -> (B N) L M")

            y_pred_tf, y_pred = outputs[0], outputs_notf[0]

            y_displacements = torch.concatenate([x[:, -1:], y], axis=1).diff(axis=1).cpu().numpy()
            y_pred_displacements = torch.concatenate([x[:, -1:], y_pred_tf], axis=1).diff(axis=1).cpu().numpy()
            self.plot_histogram(y_displacements[:, :, 0].flatten(), y_pred_displacements[:, :, 0].flatten(), "x")
            self.plot_histogram(y_displacements[:, :, 1].flatten(), y_pred_displacements[:, :, 1].flatten(), "y")

            x = self.upsample(x)
            y = self.upsample(y)
            y_pred_tf = self.upsample(y_pred_tf)
            y_pred = self.upsample(y_pred)

            x = x.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            y_pred_tf = y_pred_tf.detach().cpu().numpy()
            y_pred = y_pred.detach().cpu().numpy()

            self.plot_trajectory(x[0], y[0], y_pred_tf[0], memo="teacher forcing")
            self.plot_trajectory(x[0], y[0], y_pred[0], memo="no teacher forcing")

            self.plot_traj_on_pitch(x[:22], y[:22], y_pred[:22])

    def test_step(self, batch, batch_idx):
        x, y = batch

        y_pred = self.roll_out(
            self.downsample(x),
            n_steps=self.downsample(y).shape[1],
        )
        y_pred = self.upsample(y_pred)

        if not self.single_agent:
            x = rearrange(x, "B L N M -> (B N) L M")
            y = rearrange(y, "B L N M -> (B N) L M")
        eval_metrics = self.compute_eval_metrics(y_gt=y, y_pred=y_pred)

        for name, value in eval_metrics.items():
            self.log(name, value, prog_bar=False)

        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()

        memo = f"test[{batch_idx}]_rmse_100pct={eval_metrics['rmse_100pct']:.2f}"
        self.plot_traj_on_pitch(x[:22], y[:22], y_pred[:22], memo=memo)


# def mdn_nll_loss(y, pi, mu, sigma):
#     batch_size, seq_len, _ = y.shape
#     n_mixtures = mu.shape[-1] // 2

#     # Reshape mu and sigma
#     mu = mu.view(batch_size, seq_len, n_mixtures, 2)
#     sigma = sigma.view(batch_size, seq_len, n_mixtures, 2)

#     m = torch.distributions.Normal(loc=mu, scale=sigma)
#     y_expanded = y.unsqueeze(2).expand(batch_size, seq_len, n_mixtures, 2)
#     log_prob = m.log_prob(y_expanded)
#     log_prob = log_prob.sum(dim=-1) + torch.log(pi)

#     return -torch.logsumexp(log_prob, dim=2).mean()
# Importing the necessary libraries
