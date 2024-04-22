from soccertrack.datasets.datasets.trajectory_datamodule import (
    TrajectoryDataModule,
)
from pytorch_lightning import Trainer
from pathlib import Path
from soccertrack.motion_model.models.lstm import SingleTargetLSTM, MultiTargetLSTM
from pytorch_lightning.callbacks import LearningRateMonitor
import sys

sys.path.append("/Users/atom/Github/TeamTrack")
from teamtrack.base_module import BaseMotionModule

if __name__ == "__main__":
    trajectory_dataset_path = Path(
        "/Users/atom/Github/TeamTrack/data/trajectory_dataset"
    )

    data_dir = trajectory_dataset_path / "F_Soccer_Tsukuba3"
    dm = TrajectoryDataModule(data_dir, single_agent=True)
    # dm = TrajectoryDataModule(data_dir, single_agent=False)
    dm.setup()

    lstm = SingleTargetLSTM(16, n_layers=2, dropout=0.2)
    # lstm = MultiTargetLSTM(44, 64, n_layers=3, dropout=0.2)
    module = BaseMotionModule(model=lstm, roll_out_steps=144)

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    trainer = Trainer(
        accelerator="cpu",
        enable_checkpointing=False,
        max_epochs=100,
        callbacks=[lr_monitor],
    )

    trainer.fit(module, dm)
    trainer.test(module, dm)
