import re
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, default_collate
from torchvision.transforms import ToTensor, Compose
from einops import rearrange
from functools import partial
from teamtrack.transforms import SportSpecificTransform, Noise

# def single_agent_collate_fn(batch):
#     x = torch.Tensor([seq for item in batch for seq in rearrange(item[0], "L N D ->  N L D")])
#     y = torch.Tensor([seq for item in batch for seq in rearrange(item[1], "L N D ->  N L D")])
#     # x, y = default_collate(batch)

#     # B: batch size, L: sequence length, N: number of agents, D: dimension
#     # x = rearrange(x, "B L N D -> (B N) L D")
#     # y = rearrange(y, "B L N D -> (B N) L D")

#     return x, y




def multi_agent_collate_fn(batch, max_num_agents, dummy_value=-1000):
    batch_size = len(batch)
    x, y = [], []
    for i, (x_seq, y_seq) in enumerate(batch):
        x_shape = (x_seq.shape[0], max_num_agents, x_seq.shape[-1])
        y_shape = (y_seq.shape[0], max_num_agents, y_seq.shape[-1])

        x_full = torch.full(x_shape, dummy_value, dtype=x_seq.dtype, device=x_seq.device)
        y_full = torch.full(y_shape, dummy_value, dtype=y_seq.dtype, device=y_seq.device)

        num_agents_x = min(x_seq.shape[1], max_num_agents)
        num_agents_y = min(y_seq.shape[1], max_num_agents)

        x_full[:, :num_agents_x, :] = x_seq[:, :num_agents_x, :]
        y_full[:, :num_agents_y, :] = y_seq[:, :num_agents_y, :]

        x.append(x_full)
        y.append(y_full)
    return torch.stack(x), torch.stack(y)


def smooth_sequence(sequence, window_size=21):
    sequence = torch.tensor(sequence, dtype=torch.float32) if not isinstance(sequence, torch.Tensor) else sequence
    smoothed_sequence = torch.zeros_like(sequence)

    for i in range(sequence.shape[0]):
        start = max(0, i - window_size + 1)
        end = i + 1
        smoothed_sequence[i] = torch.mean(sequence[start:end], dim=0)

    return smoothed_sequence


class TrajectoryDataset(Dataset):
    def __init__(self, data_dir, transform=None, flatten=False, split=50, add_noise=False):
        self.flatten = flatten
        self.transform = transform
        self.files = self.get_files(data_dir)
        self.split = split
        self.add_noise = True
        self.noise = Noise(mean=0, std=0.1)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # data is of shape (seq_len, num_agents * 2)
        data = self.load_data(self.files[idx])

        # Reshape data to (seq_len, num_players, 2)
        n_players = data.shape[1] // 2
        x_player_indices = np.arange(0, n_players * 2, 2)
        y_player_indices = np.arange(1, n_players * 2, 2)
        data = np.stack([data[:, x_player_indices], data[:, y_player_indices]], axis=-1)
        assert data.shape[1] == n_players, f"Expected {n_players} players, got {data.shape[1]}({data.shape})"
        assert data.shape[2] == 2, f"Expected 2 dimensions, got {data.shape[2]}({data.shape})"

        if self.transform:
            data = self.transform(data)

        out_data = data[: self.split]
        out_label = data[self.split :]
        return out_data, out_label

    def load_data(self, path):
        data = np.loadtxt(path, delimiter=",")
        return data

    def get_files(self, data_dir):
        files = []
        for file in data_dir.glob("*.txt"):
            files.append(file)
        return files


def random_ordering(data):
    # randomize and flatten the agent axis
    num_agents = data.shape[1]
    data = data[:, torch.randperm(num_agents), :]
    return data


class TrajectoryDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "path/to/dir",
        batch_size: int = 32,
        pin_memory: bool = False,
        num_workers: int = 1,
        shuffle: bool = True,
        single_agent=False,
        smooth=True,
        split=96,
        max_num_agents=None,
        sport="Soccer",
        add_noise=False,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.check_data_dir()

        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.single_agent = single_agent
        self.split = split
        self.max_num_agents = max_num_agents
        self.sport = sport
        self.add_noise = add_noise

        transforms = [torch.Tensor]
        if smooth:
            transforms.append(smooth_sequence)
        if sport:
            print(f"Using `SportsSpecificTransform` for {sport}")
            transforms.append(SportSpecificTransform(sport))
        if not single_agent:
            transforms.append(random_ordering)

        self.transform = Compose(transforms)

    def check_data_dir(self):
        # do necessary checks (datafolder structure etc.) here
        pass

    def setup(self, stage: str = "both"):
        data_dir = self.data_dir
        train_data_dir = data_dir / "train"
        val_data_dir = data_dir / "val"
        test_data_dir = data_dir / "test"

        if stage == "fit" or stage == "both":
            self.trainset = TrajectoryDataset(train_data_dir, transform=self.transform, split=self.split, add_noise=self.add_noise)
            self.valset = TrajectoryDataset(val_data_dir, transform=self.transform, split=self.split, add_noise=self.add_noise)
            self.testset = TrajectoryDataset(test_data_dir, transform=self.transform, split=self.split)
            print("trainset:", len(self.trainset))
            print("valset:", len(self.valset))
            print("testset:", len(self.testset))
        if stage == "test" or stage == "both":
            self.testset = TrajectoryDataset(test_data_dir, transform=self.transform, split=self.split)

    def train_dataloader(self):
        collate_fn = single_agent_collate_fn if self.single_agent else partial(multi_agent_collate_fn, max_num_agents=self.max_num_agents)
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        collate_fn = single_agent_collate_fn if self.single_agent else partial(multi_agent_collate_fn, max_num_agents=self.max_num_agents)
        return DataLoader(
            self.testset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        collate_fn = single_agent_collate_fn if self.single_agent else partial(multi_agent_collate_fn, max_num_agents=self.max_num_agents)
        return DataLoader(
            self.testset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )


if __name__ == "__main__":
    dataset_name = "F_Soccer_Tsukuba3"
    data_dir = f"/groups/gaa50073/atom/SoccerTrackProject/SoccerTrack_Data/teamtrack/data/{dataset_name}"
    dm = TrajectoryDataModule(data_dir)
