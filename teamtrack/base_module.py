import wandb
import pytorch_lightning as pl
import torch
from torchmetrics.functional import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
from teamtrack.transforms import SportSpecificTransform, Resample
from mplsoccer import Pitch


class BaseMotionModule(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-3, roll_out_steps=10, single_agent=False, sport="soccer", original_fps=25, inference_fps=25):
        super().__init__()

        self.learning_rate = learning_rate
        self.roll_out_steps = roll_out_steps
        self.single_agent = single_agent
        self.model = model
        self.sport = sport
        self.inverse_norm = SportSpecificTransform(sport).inverse_transform
        self.downsample = Resample(original_fps, inference_fps)
        self.upsample = Resample(inference_fps, original_fps)
        self.save_hyperparameters()
        print(model)

    def forward(self, x):
        return self.model(x)

    def roll_out(self, x, n_steps=None, y_gt=None):
        return self.model.roll_out(x, n_steps=n_steps, y_gt=y_gt)

    def compute_eval_metrics(self, y_gt, y_pred):
        total_steps = self.roll_out_steps
        eval_steps = [0, int(0.1 * total_steps), int(0.5 * total_steps), total_steps - 1]
        eval_step_names = ["first_step", "10pct", "50pct", "100pct"]
        eval_metrics = {}

        for step, name in zip(eval_steps, eval_step_names):
            rmse = torch.sqrt(mean_squared_error(y_pred[:, step], y_gt[:, step]))
            eval_metrics[f"rmse_{name}"] = rmse
        return eval_metrics

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.roll_out(x, n_steps=y.shape[1])
        eval_metrics = self.compute_eval_metrics(y_gt=y, y_pred=y_pred)

        for name, value in eval_metrics.items():
            self.log(name, value, prog_bar=False)

        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()

        memo = f"test[{batch_idx}]_rmse_100pct={eval_metrics['rmse_100pct']:.2f}"
        self.plot_traj_on_pitch(x[:22], y[:22], y_pred[:22], memo=memo)

    # def training_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_pred = self.roll_out(
    #         x,
    #         y_gt=y,
    #     )
    #     loss = self.compute_loss(y_gt=y, y_pred=y_pred)
    #     self.log("train_loss", loss, prog_bar=True)
    #     return loss

    # def validation_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_pred_tf = self.roll_out(x, y_gt=y)
    #     loss_tf = self.compute_loss(y_gt=y, y_pred=y_pred_tf)
    #     self.log("val_loss_tf", loss_tf, prog_bar=False)

    #     y_pred = self.roll_out(x)
    #     loss = self.compute_loss(y_gt=y, y_pred=y_pred)
    #     self.log("val_loss", loss, prog_bar=True)

    #     if batch_idx == 0:
    #         self._log_validation_plots(x, y, y_pred_tf, y_pred)

    def _log_validation_plots(self, x, y, y_pred_tf, y_pred):
        y_displacements = torch.concatenate([x[:, -1:], y], axis=1).diff(axis=1).cpu().numpy()
        y_pred_displacements = torch.concatenate([x[:, -1:], y_pred_tf], axis=1).diff(axis=1).cpu().numpy()
        self.plot_histogram(y_displacements[:, :, 0].flatten(), y_pred_displacements[:, :, 0].flatten(), "x")
        self.plot_histogram(y_displacements[:, :, 1].flatten(), y_pred_displacements[:, :, 1].flatten(), "y")

        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        y_pred_tf = y_pred_tf.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()

        self.plot_trajectory(x[0], y[0], y_pred_tf[0], memo="teacher forcing")
        self.plot_trajectory(x[0], y[0], y_pred[0], memo="no teacher forcing")

    def plot_trajectory(self, x, y, y_pred, memo=""):
        current_epoch = self.current_epoch
        title = f"Trajectory at epoch {current_epoch} ({memo})"
        fig = plt.figure(figsize=(10, 10))

        plt.scatter(x[:, 0], x[:, 1], label="input")
        plt.scatter(y[:, 0], y[:, 1], label="ground truth")
        plt.scatter(y_pred[:, 0], y_pred[:, 1], label="prediction")
        plt.title(title)
        plt.legend()

        im = plot_to_image(fig)
        self.logger.experiment.log({f"Trajectory ({memo})": [wandb.Image(im)]}, step=self.global_step)
        plt.close(fig)

    def plot_histogram(self, y_displacements, y_pred_displacements, tag):
        current_epoch = self.current_epoch

        bins = np.linspace(-1, 1, 300)
        title = f"{tag} displacements at epoch {current_epoch}"
        plt.figure(figsize=(10, 10))

        plt.hist(y_displacements, label="ground truth", bins=bins)
        plt.hist(y_pred_displacements, label="prediction", bins=bins)
        plt.title(title)
        plt.legend()

        self.logger.experiment.log({f"{tag} displacements": [wandb.Image(plt)]}, step=self.global_step)
        plt.close()

    def plot_traj_on_pitch(self, x, y, y_pred, memo=""):
        # x, y, y_pred should be in meters and shape (22, n_steps, 2)
        assert x.shape[0] == 22 and x.shape[2] == 2, f"Expected shape (22, n_steps, 2), got {x.shape}"
        current_epoch = self.current_epoch
        # Initialize the pitch with custom dimensions and background color
        pitch = Pitch(
            pitch_length=105,
            pitch_width=68,
            pitch_color="black",
            line_color="white",
        )

        title = f"Trajectory on pitch at epoch {current_epoch} ({memo})"
        fig, ax = pitch.draw()

        for i in range(22):
            if i == 0:
                ax.plot(x[i, :, 0], x[i, :, 1], color="cyan", linestyle="-", linewidth=2, label="Observation", alpha=0.9)
                ax.plot(y[i, :, 0], y[i, :, 1], color="yellow", linestyle="--", linewidth=2, label="Ground Truth", alpha=0.9)
                ax.plot(y_pred[i, :, 0], y_pred[i, :, 1], color="magenta", linestyle="-", linewidth=1.5, label="Prediction", alpha=0.8)
            else:
                ax.plot(x[i, :, 0], x[i, :, 1], color="cyan", linestyle="-", linewidth=2, alpha=0.9)
                ax.plot(y[i, :, 0], y[i, :, 1], color="yellow", linestyle="--", linewidth=2, alpha=0.9)
                ax.plot(y_pred[i, :, 0], y_pred[i, :, 1], color="magenta", linestyle="-", linewidth=1.5, alpha=0.8)
        plt.title(title)
        plt.legend()

        im = plot_to_image(fig)
        self.logger.experiment.log({f"Trajectory on pitch ({memo})": [wandb.Image(im)]}, step=self.global_step)
        plt.close(fig)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.3, patience=10, min_lr=1e-8)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_nll_loss"}


def plot_to_image(figure):
    buf = io.BytesIO()
    figure.savefig(buf, format="png")
    buf.seek(0)
    image = Image.open(buf)
    return image
