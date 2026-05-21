#!/usr/bin/env python3
"""
Test.py
----------
Load ALL checkpoints from a given run and run inference on future Meteostat data. 
Run analysis on the inference.
"""
import warnings
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Suppress noisy PyTorch warnings
def suppress_runtime_warnings():
    """
    Suppress known non-critical runtime warnings from NumPy, PyTorch, and pandas.
    Keeps logs clean while avoiding suppression of meaningful errors.
    """

    # --- NumPy 1.x → 2.x compatibility warnings ---
    warnings.filterwarnings(
        "ignore",
        message=r".*compiled using NumPy 1\.x.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*A module that was compiled using NumPy 1\.x.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*Some module may need to rebuild instead.*",
        category=UserWarning,
    )

    # --- PyTorch NumPy init warnings ---
    warnings.filterwarnings(
        "ignore",
        message=r".*Failed to initialize NumPy: _ARRAY_API not found.*",
        category=UserWarning,
        module="torch"
    )

    # --- Pandas SettingWithCopyWarning spam ---
    warnings.filterwarnings(
        "ignore",
        category=pd.errors.SettingWithCopyWarning
    )

    # --- Optional: silence all UserWarnings (comment out if debugging) ---
    # warnings.filterwarnings("ignore", category=UserWarning)

    logging.info("🔇 Non-critical runtime warnings suppressed.")
suppress_runtime_warnings()
# ===========================================================
import argparse
import os
import re
import json
import sys
from datetime import datetime
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar
from meteostat import Point
from DataPipelineWorkShop import (get_hourly_example, 
                                  PreprocessingContext, 
                                  ForecastContext, 
                                  ModelContext, 
                                  TrainingContext, 
                                  ExperimentContext, 
                                  make_dataloaders)
from VanillaTransformer import MeteoVanillaTransformerEncoder, CloserType
from Informer import MeteoInformerHourglassTransformer
# ===========================================================
import matplotlib.pyplot as plt
sys.path.append('Utils/')
from PlotUtils import setMplParam, getColour, getHistoParam 
setMplParam()
# ===========================================================
def parse_args():
    p = argparse.ArgumentParser("Forecast future Meteo data using a trained Transformer")
    p.add_argument("--gpu", nargs="+", type=int, default=[], help="List of GPU IDs to use")
    p.add_argument("--date", type=str, required=True, help="Run date: YYYYMMDD")
    p.add_argument("--time", type=str, required=True, help="Run time: HHMMSS")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--log_dir", type=str, default="logs", help="Base logging directory")

    return vars(p.parse_args())

# ===========================================================
def lock_and_load(config):
    """Detect if GPU should be used and optionally set cuda device. Returns gpu_enabled flag."""
    gpu_available = torch.cuda.is_available()
    logging.info(f"torch.cuda.is_available(): {gpu_available}")
    available_devices = list(range(torch.cuda.device_count()))
    logging.info(f"Available CUDA devices: {available_devices}")

    if gpu_available and config["gpu"]:
        selected_gpu = config["gpu"][0]

        if selected_gpu in available_devices:
            torch.cuda.set_device(selected_gpu)
            torch.cuda.empty_cache()
            torch.set_float32_matmul_precision("highest")
            logging.info(f"🔥 Using GPU: cuda:{selected_gpu}")
            return True
        else:
            logging.info(f"⚠️ GPU {selected_gpu} not found. Falling back to CPU.")
            return False

    logging.info("CUDA not available. Using CPU.")
    return False

# ===========================================================
def get_contexts(config, log_dir):
    ctx_json_path = os.path.join(
        log_dir,
        f"{config['date']}_{config['time']}_context.json"
    )

    with open(ctx_json_path, "r") as f:
        ctx_dict = json.load(f)

    pre_ctx   = PreprocessingContext(**ctx_dict["preprocessing"])
    fc_ctx    = ForecastContext(**ctx_dict["forecast"])
    model_ctx = ModelContext(**ctx_dict["model"])
    exp_ctx = ExperimentContext(
        preprocessing=pre_ctx,
        forecast=fc_ctx,
        model=model_ctx
    )
    target_features = ctx_dict["target_features"]
    # logging.info(f"Target features: {target_features}")
    closer_type = CloserType.from_string(ctx_dict["closer_type"])
    first_year = ctx_dict["first_year"]
    last_year = ctx_dict["last_year"]
    return exp_ctx, target_features, closer_type, first_year, last_year

# ===========================================================
def get_input_features_from_checkpoint(ckpt_path):
    ckpt = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
    )
    hparams = ckpt.get("hyper_parameters", {})

    if "input_features" not in hparams:
        raise KeyError(
            "Checkpoint does not contain 'input_features'. "
            "For old checkpoints, save the training feature list into context.json."
        )

    return hparams["input_features"]

# ===========================================================
def get_checkpoints(config):
    # path to checkpoints dir
    ckpt_dir = os.path.join(
        config["log_dir"],
        config["date"],
        config["time"],
        "checkpoints",
    )

    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(f"checkpoint directory not found: {ckpt_dir}")

    ckpts = sorted([os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")])
    if not ckpts:
        raise FileNotFoundError(f"no .ckpt files found in {ckpt_dir}")

    logging.info(f"found {len(ckpts)} checkpoints")
    for f in ckpts:
        logging.info(f" - {f}")
    return ckpts
# ===========================================================
# ===========================================================
# ===========================================================
def compute_metrics_from_residuals(residual_df: pd.DataFrame):
    if residual_df.empty:
        return {
            "n_points": 0,
            "mse": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
            "bias": float("nan"),
            "residual_std": float("nan"),
        }

    mse = residual_df["squared_error"].mean()
    rmse = mse ** 0.5
    mae = residual_df["abs_error"].mean()
    bias = residual_df["residual"].mean()
    residual_std = residual_df["residual"].std()

    metrics = {
        "n_points": len(residual_df),
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "residual_std": residual_std,
    }

    # Per-lead-time RMSE: rmse_lead_01, ..., rmse_lead_12
    lead_mse = residual_df.groupby("lead_time")["squared_error"].mean()
    for lead, val in lead_mse.items():
        metrics[f"rmse_lead_{int(lead):02d}"] = val ** 0.5

    # Per-feature RMSE
    feature_mse = residual_df.groupby("target_feature")["squared_error"].mean()
    for feature, val in feature_mse.items():
        metrics[f"rmse_{feature}"] = val ** 0.5

    return metrics
# ===========================================================
def evaluate_checkpoint(model,   # LightningModule
                        test_dl, # torch.utils.data.DataLoader
                        exp_ctx: ExperimentContext, 
                        checkpoint_name: str, 
                        device) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    model.to(device)
    model.eval()

    horizon = exp_ctx.forecast.horizon
    target_features = model.get_target_features()

    residual_rows = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dl):
            x, y_true, x_mask, y_mask = batch

            x = x.to(device)
            y_true = y_true.to(device)
            x_mask = x_mask.to(device)
            y_mask = y_mask.to(device)

            y_pred = model(x, mask=x_mask.any(-1))

            idx = model.closer.target_indices

            # final horizon only
            y_pred_f = y_pred[:, -horizon:, :]          # (B, H, D_target)
            y_true_f = y_true[:, -horizon:, idx]        # (B, H, D_target)
            y_mask_f = y_mask[:, -horizon:, idx]        # (B, H, D_target)

            residual = y_pred_f - y_true_f

            B, H, D = residual.shape

            for b in range(B):
                window_id = batch_idx * test_dl.batch_size + b

                for h in range(H):
                    lead_time = h + 1

                    for d, feature in enumerate(target_features):
                        if not bool(y_mask_f[b, h, d]):
                            continue

                        true_val = float(y_true_f[b, h, d].detach().cpu())
                        pred_val = float(y_pred_f[b, h, d].detach().cpu())
                        res_val = pred_val - true_val

                        residual_rows.append({
                            "checkpoint": checkpoint_name,
                            "window_id": window_id,
                            "lead_time": lead_time,
                            "target_feature": feature,
                            "y_true": true_val,
                            "y_pred": pred_val,
                            "residual": res_val,
                            "abs_error": abs(res_val),
                            "squared_error": res_val ** 2,
                        })

    residual_df = pd.DataFrame(residual_rows)

    metrics = compute_metrics_from_residuals(residual_df)
    metrics["checkpoint"] = checkpoint_name

    return metrics, residual_df
# ===========================================================
def save_checkpoint_metrics(all_metrics, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    metrics_df = pd.DataFrame(all_metrics)
    out_path = os.path.join(out_dir, "checkpoint_test_metrics.csv")
    metrics_df.to_csv(out_path, index=False)

    logging.info(f"✅ Checkpoint metrics saved to: {out_path}")
    return out_path
# ===========================================================
def save_residual_tables_by_checkpoint(all_residual_dfs, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    if not all_residual_dfs:
        logging.warning("No residual tables to save.")
        return []

    out_paths = []

    for residual_df in all_residual_dfs:
        if residual_df.empty:
            continue

        checkpoint_name = residual_df["checkpoint"].iloc[0]
        safe_name = checkpoint_name.replace("/", "_").replace("\\", "_")

        out_path = os.path.join(
            out_dir,
            f"{safe_name}_test_residuals.csv"
        )

        # checkpoint column is now redundant because the filename contains it
        residual_df = residual_df.drop(columns=["checkpoint"])

        residual_df.to_csv(out_path, index=False)
        out_paths.append(out_path)

        logging.info(f"✅ Residual table saved to: {out_path}")

    return out_paths
# ===========================================================

# ===========================================================
def build_model_signature(exp_ctx: ExperimentContext) -> str:
    """
    Build a short hyperparameter signature string from the context.
    LR is intentionally left blank.
    """
    mc = exp_ctx.model
    fc = exp_ctx.forecast

    signature = (
        f"d{mc.d_model}_"
        f"h{mc.n_heads}_"
        f"ff{mc.d_ff}_"
        f"st{mc.starter_num_layers}_"
        f"cl{mc.closer_num_layers}_"
        f"win{fc.window}_"
        f"ho{fc.horizon}_"
        f"dr{mc.dropout}"
        "_"
        # f"lr"
    )
    return signature
# ===========================================================
def extract_metrics_from_trainer(trainer):
    """Extract logged RMSE values after test_step and test_epoch_end."""
    metrics = trainer.callback_metrics
    results = {}

    for key, val in metrics.items():
        if "rmse" in key:
            results[key] = float(val.cpu())

    return results
# ===========================================================
def extract_val_loss(ckpt_path):
    """
    Extract val_loss from checkpoint filename.
    Returns float('inf') if not found.
    """
    fname = os.path.basename(ckpt_path)
    match = re.search(r"val_loss[=\-]?([0-9]*\.[0-9]+)", fname)
    if match:
        return float(match.group(1))
    return float("inf")

# ===========================================================
def find_best_checkpoint(ckpt_list):
    """
    Return the checkpoint path with the smallest validation loss.
    """
    best_ckpt = None
    best_loss = float("inf")

    for ckpt in ckpt_list:
        val_loss = extract_val_loss(ckpt)
        if val_loss < best_loss:
            best_loss = val_loss
            best_ckpt = ckpt

    return best_ckpt, best_loss

# ===========================================================
def run():
    config = parse_args()
    is_lock_and_loaded = lock_and_load(config)
    date_time_dir = os.path.join(config["log_dir"], config["date"], config["time"])
    os.makedirs(date_time_dir, exist_ok=True)

    # 1. Load experiment context
    exp_ctx, _, closer_type, first_year, last_year = get_contexts(config, date_time_dir)
    
    # 2. Prepare the same dataset used during training
    kbh = Point(lat=55.6761, lon=12.5683)

    df_raw = get_hourly_example(
        kbh,
        start=datetime(first_year, 1, 1),
        end=datetime(last_year, 12, 31),
    )

    _, _, test_dl = make_dataloaders(
        df_raw,
        exp_ctx,
        batch_size=config["batch_size"],
        num_workers=2,
    )

    # 3. Gather checkpoints
    ckpts = get_checkpoints(config)
    
    # 4. Start output table
    all_metrics = []
    all_residual_dfs = []

    device = torch.device(
        f"cuda:{config['gpu'][0]}"
        if is_lock_and_loaded and config["gpu"]
        else "cpu"
    )
    
    # 5. Loop over checkpoints
    for ckpt in ckpts:
        ckpt_name = os.path.basename(ckpt).replace("=", "-").replace(".ckpt", "")
        logging.info(f"🔍 Testing checkpoint: {ckpt_name}")

        input_features = get_input_features_from_checkpoint(ckpt)
        model = MeteoVanillaTransformerEncoder.load_from_checkpoint(
            checkpoint_path=ckpt,
            model_ctx=exp_ctx.model,
            forecast_ctx=exp_ctx.forecast,
            input_features=input_features,
            closer_type=closer_type,
            training_ctx=None
        )
        
        # model = MeteoInformerHourglassTransformer.load_from_checkpoint(
        #     checkpoint_path=ckpt,
        #     model_ctx=exp_ctx.model,
        #     forecast_ctx=exp_ctx.forecast,
        #     input_features=input_features,
        #     closer_type=closer_type,
        #     training_ctx=None
        # )

        metrics, residual_df = evaluate_checkpoint(
            model=model,
            test_dl=test_dl,
            exp_ctx=exp_ctx,
            checkpoint_name=ckpt_name,
            device=device,
        )

        all_metrics.append(metrics)
        all_residual_dfs.append(residual_df)

        logging.info(f"📊 Metrics: {metrics}")

    # 6. Save as single CSV
    out_dir = os.path.join("test_results", config["date"], config["time"])
    save_checkpoint_metrics(all_metrics, out_dir)
    save_residual_tables_by_checkpoint(all_residual_dfs, out_dir)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    suppress_runtime_warnings()
    logging.basicConfig(level=logging.INFO)
    run()
