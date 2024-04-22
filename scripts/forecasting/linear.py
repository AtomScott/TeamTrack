from soccertrack.datasets.datasets.trajectory_datamodule import (
    TrajectoryDataModule,
)
from pytorch_lightning import Trainer
from pathlib import Path
from soccertrack.motion_model.models.linear import SingleTargetLinear

import sys

sys.path.append("/Users/atom/Github/TeamTrack")
from teamtrack.base_module import BaseMotionModule

if __name__ == "__main__":
    trajectory_dataset_path = Path(
        "/Users/atom/Github/TeamTrack/data/trajectory_dataset"
    )

    data_dir = trajectory_dataset_path / "F_Soccer_Tsukuba3"
    dm = TrajectoryDataModule(data_dir, single_agent=True, batch_size=32)
    dm.setup()

    print(dm.trainset[0][0].shape)

    module = BaseMotionModule(model=SingleTargetLinear(96), roll_out_steps=144)

    trainer = Trainer(accelerator="cpu", enable_checkpointing=False)

    trainer.test(module, dm)
