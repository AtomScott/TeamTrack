from soccertrack.datasets.datasets.trajectory_datamodule import (
    TrajectoryDataModule,
)
from pytorch_lightning import Trainer
from soccertrack.motion_model.base_module import BaseMotionModule
from pathlib import Path
from soccertrack.motion_model.models.gnn import MultiTargetGNN


if __name__ == "__main__":
    trajectory_dataset_path = Path(
        "/Users/atom/Github/TeamTrack/data/trajectory_dataset"
    )

    data_dir = trajectory_dataset_path / "F_Soccer_Tsukuba3"
    dm = TrajectoryDataModule(data_dir, single_agent=False, split=96)
    dm.setup()

    gnn = MultiTargetGNN(
        input_channels=96 * 2,
        hidden_channels=64,
        output_channels=2,
        n_layers=3,
        dropout=0.2,
        use_complete_graph=True,
    )
    module = BaseMotionModule(model=gnn, roll_out_steps=144)

    trainer = Trainer(accelerator="cpu", enable_checkpointing=False, max_epochs=100)

    trainer.fit(module, dm)
    trainer.test(module, dm)
