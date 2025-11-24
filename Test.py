#!/usr/bin/env python3
"""
Predict.py
----------
Load ALL checkpoints from a given run and run inference on future Meteostat data.
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
from meteostat import Point
from DataPipelineWorkShop import get_hourly_example, PreprocessingContext, ForecastContext, ModelContext, ExperimentContext, make_dataloaders, make_single_window_dataframe, MeteoPreprocessor
from VanillaTransformer import MeteoVanillaTransformerEncoder
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
    p.add_argument("--batch_size", type=int, default=128)
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
    target_features = ctx_dict["target_features"]

    exp_ctx = ExperimentContext(
        preprocessing=pre_ctx,
        forecast=fc_ctx,
        model=model_ctx
    )
    return exp_ctx, target_features

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
    exp_ctx, target_features = get_contexts(config, date_time_dir)
    logging.info(f"Target features: {target_features}")

    # 2. Prepare dataset
    kbh = Point(lat=55.6761, lon=12.5683)
    df_raw = get_hourly_example(kbh, start=datetime(1988, 1, 1), end=datetime(2018, 12, 31))
    _, _, test_dl = make_dataloaders(
        df_raw, exp_ctx,
        batch_size=config["batch_size"],
        num_workers=2
    )

    input_features = test_dl.dataset._get_available_features()

    # 3. Gather checkpoints
    ckpts = get_checkpoints(config)

    # 4. Start output table
    all_results = []

    # 5. Loop over checkpoints
    for ckpt in ckpts:
        ckpt_name = os.path.basename(ckpt).replace("=", "-").replace(".ckpt", "")

        logging.info(f"🔍 Testing checkpoint: {ckpt_name}")

        model = MeteoVanillaTransformerEncoder.load_from_checkpoint(
            ckpt,
            model_ctx=exp_ctx.model,
            forecast_ctx=exp_ctx.forecast,
            input_features=input_features,
        )
        model.eval()

        trainer = Trainer(
            accelerator="gpu" if is_lock_and_loaded else "cpu",
            devices=config["gpu"] if is_lock_and_loaded and config["gpu"] else 1,
            logger=False,
        )

        trainer.test(model, dataloaders=test_dl, verbose=False)

        metrics = extract_metrics_from_trainer(trainer)
        metrics["checkpoint"] = ckpt_name
        all_results.append(metrics)

        logging.info(f"📊 Metrics: {metrics}")

    # 6. Save as single CSV
    results_df = pd.DataFrame(all_results)

    out_dir = os.path.join("test_results", config["date"], config["time"])
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "checkpoint_test_metrics.csv")
    results_df.to_csv(out_path, index=False)

    logging.info(f"✅ All checkpoint metrics saved to:\n{out_path}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    suppress_runtime_warnings()
    logging.basicConfig(level=logging.INFO)
    run()
