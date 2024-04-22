from soccertrack.datasets.datasets.trajectory_datamodule import TrajectoryDataModule
from pytorch_lightning.cli import LightningCLI

import sys

sys.path.append("/Users/atom/Github/TeamTrack")
from teamtrack.base_module import BaseMotionModule


def cli_main():
    cli = LightningCLI(BaseMotionModule, TrajectoryDataModule)


if __name__ == "__main__":
    cli_main()
