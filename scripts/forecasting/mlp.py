from soccertrack.datasets.datasets.trajectory_datamodule import (
    TrajectoryDataModule,
)
from pytorch_lightning import Trainer
from soccertrack.motion_model.base_module import BaseMotionModule
from pathlib import Path
from soccertrack.motion_model.models.mlp import SingleTargetMLP


if __name__ == "__main__":
    trajectory_dataset_path = Path(
        "/Users/atom/Github/TeamTrack/data/trajectory_dataset"
    )

    data_dir = trajectory_dataset_path / "F_Soccer_Tsukuba3"
    dm = TrajectoryDataModule(data_dir, single_agent=True)
    dm.setup()

    mlp = SingleTargetMLP(
        input_dim=96 * 2, hidden_dims=[50, 50, 50], output_dim=2, dropout_prob=0.2
    )
    module = BaseMotionModule(model=mlp, roll_out_steps=144)

    trainer = Trainer(accelerator="cpu", enable_checkpointing=False, max_epochs=10)

    trainer.fit(module, dm)
    trainer.test(module, dm)
    
