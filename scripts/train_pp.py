import os
from pathlib import Path

import torch
import wandb
from pytorch_lightning import Trainer, callbacks, seed_everything
from pytorch_lightning.loggers import WandbLogger

from sigmadock.core.callbacks import EMAWithRampup, FullNaNCheckCallback
from sigmadock.data_pp import PPDataModule
from sigmadock.diff.denoiser import SigmaDockDenoiser
from sigmadock.net.model import EquiformerV2
from sigmadock.oracle import HPARAMS
from sigmadock.trainer import SigmaLightningModule

# ------------------------------------------------------------------ #
# Hardcoded paths
# ------------------------------------------------------------------ #
# CSV_PATH = "/p/project1/profound/al-zeqri1/DiffDock-PP/datasets/DIPS/data_file.csv"
CSV_PATH= "/e/home/jusers/al-zeqri1/jupiter/complexa/sigmadock/datasets/data_file_debug.csv"
ROOT_DIR = "/e/home/jusers/al-zeqri1/jupiter/complexa/sigmadock/datasets/DIPS/pairs_pruned/DIPS/data/DIPS/interim/pairs-pruned"
EXP_DIR  = Path("/e/home/jusers/al-zeqri1/jupiter/complexa/sigmadock/experiments/pp_run_001")
# ------------------------------------------------------------------ #

def main() -> None:
    seed_everything(42, workers=True)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_dir  = EXP_DIR / "checkpoints"
    wandb_dir = EXP_DIR / "wandb_logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    wandb_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # DataModule
    # ------------------------------------------------------------------ #
    datamodule = PPDataModule(
        csv_path=CSV_PATH,
        root_dir=ROOT_DIR,
        batch_size=4,
        num_workers=4,
        coordinate_noise=0.1,
    )

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    equimodel = EquiformerV2(
        use_esm_embeddings=False,
        num_layers=4,
        num_heads=4,
        atom_feature_dims=[118, 6, 3, 6, 6, 6, 2, 2, 5, 3],
        average_degrees=HPARAMS.all_degrees,
        edge_feature_dims=[5, 2, 2, 4],
        lmax_list=[2],
        mmax_list=[2],
        protein_ligand_interactions=True,
        ligand_ligand_interactions=False,
        sphere_channels=128,
        edge_channels=128,
        attn_hidden_channels=64,
        attn_alpha_channels=64,
        attn_value_channels=64,
        ffn_hidden_channels=128,
        t_emb_dim=32,
        t_emb_type="fourier",
        t_emb_scale=1000,
        distance_expansion_dim=32,
        smearing_type="gaussian",
        radial_cutoff_function="gaussian",
        rel_distance=False,
        alpha_drop=0.0,
        drop_path_rate=0.0,
        zero_init_last=True,
        share_edge_mlp=False,
    )
    
    # equimodel = EquiformerV2(
    # use_esm_embeddings=False,
    # num_layers=2,
    # num_heads=2,
    # atom_feature_dims=[118, 6, 3, 6, 6, 6, 2, 2, 5, 3],
    # average_degrees=HPARAMS.all_degrees,
    # edge_feature_dims=[5, 2, 2, 4],
    # lmax_list=[1],          # was [2] — biggest memory saver
    # mmax_list=[1],          # was [2]
    # protein_ligand_interactions=True,
    # ligand_ligand_interactions=False,
    # sphere_channels=32,     # was 64
    # edge_channels=32,       # was 64
    # attn_hidden_channels=16,  # was 32
    # attn_alpha_channels=16,   # was 32
    # attn_value_channels=16,   # was 32
    # ffn_hidden_channels=32,   # was 64
    # t_emb_dim=16,             # was 32
    # t_emb_type="fourier",
    # t_emb_scale=1000,
    # distance_expansion_dim=16,  # was 32
    # smearing_type="gaussian",
    # radial_cutoff_function="gaussian",
    # rel_distance=False,
    # alpha_drop=0.0,
    # drop_path_rate=0.0,
    # zero_init_last=True,
    # share_edge_mlp=False,
    # )
    

    cache_path = EXP_DIR.parent / "cache"
    denoiser = SigmaDockDenoiser(
        equimodel,
        cache_path=cache_path,
        cutoff_complex_interactions=HPARAMS.get_edge_spec("inter_complex").r_max,
        cutoff_fragment_interactions=-1,   # no inter-fragment edges for PP
        cutoff_complex_virtual=HPARAMS.get_edge_spec("complex_lv2pv").r_max,
    )

    # ------------------------------------------------------------------ #
    # Lightning module
    # ------------------------------------------------------------------ #
    max_steps = 200_000
    lightning_model = SigmaLightningModule(
        denoiser=denoiser,
        fragment_scaling=0.5,
        trans_score_weight=1.0,
        rot_score_weight=0.5,
        trans_data_weight=0.0,
        rot_data_weight=0.0,
        max_steps=max_steps,
        num_warmup_steps=5_000,
        min_lr_start=4e-5,
        max_lr_start=3e-4,
        min_lr_end=1e-5,
        max_lr_end=8e-5,
        init_lr_start=1e-6,
        num_lr_cycles=8,
        cycle_warmup_frac=0.25,
        weight_decay=1e-4,
        optimizer_eps=1e-8,
        betas=(0.9, 0.999),
        grad_clip=1.0,
        compile=False,
    )

    # ------------------------------------------------------------------ #
    # Logger & callbacks
    # ------------------------------------------------------------------ #
    wandb_logger = WandbLogger(
        entity="sigma-dock",
        project="SigmaDock-PP",
        name="pp_run_001",
        save_dir=str(wandb_dir),
        offline=False,
        # offline=True,
    )

    all_callbacks = [
        callbacks.ModelCheckpoint(
            save_top_k=3,
            monitor="loss_val/total",
            mode="min",
            dirpath=str(ckpt_dir),
            filename="checkpoint-{step:06d}-{loss_val/total:.4f}",
            save_last=True,
        ),
        callbacks.LearningRateMonitor(logging_interval="step"),
        callbacks.ModelSummary(max_depth=3),
    ]

    # ------------------------------------------------------------------ #
    # Trainer
    # ------------------------------------------------------------------ #
    trainer = Trainer(
        max_steps=max_steps,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        # precision="bf16-mixed" if torch.cuda.is_available() else 32,
        precision= 32,
        logger=wandb_logger,
        callbacks=all_callbacks,
        log_every_n_steps=50,
        val_check_interval=1000,
        check_val_every_n_epoch=None,
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
    )

    trainer.fit(lightning_model, datamodule=datamodule)
    wandb.finish()


if __name__ == "__main__":
    main()