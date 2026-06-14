# """
# PyTorch Dataset / DataModule for protein-protein docking (DIPS format).

# Each sample is a .dill file containing two DataFrames (two protein chains).
# The longer chain is treated as the receptor (fixed); the shorter as the binder (diffused).
# """

# from __future__ import annotations

# from pathlib import Path

# import dill
# import pandas as pd
# import pytorch_lightning as pl
# import torch
# from torch.utils.data import DataLoader, Dataset

# from sigmadock.chem.pp_processing import build_pp_complex_graph


# class PPDataset(Dataset):
#     def __init__(
#         self,
#         csv_path: str | Path,
#         root_dir: str | Path,
#         split: str,
#         coordinate_noise: float | None = None,
#     ) -> None:
#         self.root_dir = Path(root_dir)
#         self.coordinate_noise = coordinate_noise

#         df = pd.read_csv(csv_path)
#         df = df[df["split"] == split].reset_index(drop=True)
#         self.samples: list[Path] = [self.root_dir / row["path"] for _, row in df.iterrows()]

#     def __len__(self) -> int:
#         return len(self.samples)

#     def __getitem__(self, idx: int):
#         path = self.samples[idx]
#         with open(path, "rb") as f:
#             raw = dill.load(f)

#         df1, df2 = raw[1], raw[2]
#         # Longer chain = receptor (fixed context); shorter = binder (diffused)
#         rec_df, bind_df = (df1, df2) if len(df1) >= len(df2) else (df2, df1)

#         # Reset indices so offsets in pp_processing are correct
#         rec_df = rec_df.reset_index(drop=True)
#         bind_df = bind_df.reset_index(drop=True)

#         data = build_pp_complex_graph(
#             rec_df,
#             bind_df,
#             coordinate_noise=self.coordinate_noise,
#         )
#         return data


# class PPDataModule(pl.LightningDataModule):
#     def __init__(
#         self,
#         csv_path: str | Path,
#         root_dir: str | Path,
#         batch_size: int = 4,
#         num_workers: int = 4,
#         coordinate_noise: float | None = None,
#     ) -> None:
#         super().__init__()
#         self.csv_path = csv_path
#         self.root_dir = root_dir
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.coordinate_noise = coordinate_noise

#     def _make_dataset(self, split: str) -> PPDataset:
#         return PPDataset(
#             csv_path=self.csv_path,
#             root_dir=self.root_dir,
#             split=split,
#             coordinate_noise=self.coordinate_noise,
#         )

#     def train_dataloader(self) -> DataLoader:
#         ds = self._make_dataset("train")
#         return DataLoader(
#             ds,
#             batch_size=self.batch_size,
#             shuffle=True,
#             num_workers=self.num_workers,
#             collate_fn=_collate,
#         )

#     def val_dataloader(self) -> DataLoader:
#         ds = self._make_dataset("val")
#         return DataLoader(
#             ds,
#             batch_size=self.batch_size,
#             shuffle=False,
#             num_workers=self.num_workers,
#             collate_fn=_collate,
#         )

#     def test_dataloader(self) -> DataLoader:
#         ds = self._make_dataset("test")
#         return DataLoader(
#             ds,
#             batch_size=self.batch_size,
#             shuffle=False,
#             num_workers=self.num_workers,
#             collate_fn=_collate,
#         )


# def _collate(batch):
#     from torch_geometric.data import Batch
#     return Batch.from_data_list(batch)


"""
PyTorch Dataset / DataModule for protein-protein docking (DIPS format).

Each sample is a .dill file containing two DataFrames (two protein chains).
The longer chain is treated as the receptor (fixed); the shorter as the binder (diffused).
"""

from __future__ import annotations

from pathlib import Path

import dill
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch

from sigmadock.chem.pp_processing import build_pp_complex_graph


class PPDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        root_dir: str | Path,
        split: str,
        coordinate_noise: float | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.coordinate_noise = coordinate_noise

        df = pd.read_csv(csv_path)
        df = df[df["split"] == split].reset_index(drop=True)
        self.samples: list[Path] = [self.root_dir / row["path"] for _, row in df.iterrows()]
        print(f"[PPDataset] split={split} | {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path = self.samples[idx]
        with open(path, "rb") as f:
            raw = dill.load(f)

        df1, df2 = raw[1], raw[2]
        rec_df, bind_df = (df1, df2) if len(df1) >= len(df2) else (df2, df1)
        rec_df = rec_df.reset_index(drop=True)
        bind_df = bind_df.reset_index(drop=True)

        return build_pp_complex_graph(
            rec_df,
            bind_df,
            coordinate_noise=self.coordinate_noise,
        )


def _collate(batch):
    return Batch.from_data_list(batch)


class PPDataModule(pl.LightningDataModule):
    def __init__(
        self,
        csv_path: str | Path,
        root_dir: str | Path,
        batch_size: int = 4,
        num_workers: int = 4,
        coordinate_noise: float | None = None,
    ) -> None:
        super().__init__()
        self.csv_path = csv_path
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.coordinate_noise = coordinate_noise

    def _make_dataset(self, split: str) -> PPDataset:
        return PPDataset(
            csv_path=self.csv_path,
            root_dir=self.root_dir,
            split=split,
            coordinate_noise=self.coordinate_noise,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._make_dataset("train"),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=_collate,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._make_dataset("val"),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=_collate,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._make_dataset("test"),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=_collate,
        )