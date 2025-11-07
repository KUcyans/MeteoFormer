#!/usr/bin/env python3
"""
Predict.py
----------
Load ALL checkpoints from a given run and run inference on future Meteostat data.
"""

import argparse
import logging
import os
import json
from datetime import datetime
import torch
from pytorch_lightning import Trainer
from meteostat import Point
from DataPipelineWorkShop import MeteoDatasetModule, get_hourly_example, PreprocessingContext, ForecastContext, ModelContext, ExperimentContext
from VanillaTransformer import MeteoVanillaTransformerEncoder
import numpy as np
import pandas as pd
import warnings

# ===========================================================
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

# ===========================================================
def parse_args():
    p = argparse.ArgumentParser("Forecast future Meteo data using a trained Transformer")

    p.add_argument("--date", type=str, required=True, help="Run date: YYYYMMDD")
    p.add_argument("--time", type=str, required=True, help="Run time: HHMMSS")
    p.add_argument("--window_size", type=int, default=24)
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--log_dir", type=str, default="./logs")

    return vars(p.parse_args())

# ===========================================================
def lock_and_load(config):
    """Set CUDA device based on config['gpu'] if available, else use CPU."""
    logging.info(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    available_devices = list(range(torch.cuda.device_count()))
    logging.info(f"Available CUDA devices: {available_devices}")

    if torch.cuda.is_available() and len(config.get("gpu", [])) > 0:
        requested_gpus = config.get("gpu", [])
        selected_gpu = int(requested_gpus[0]) if requested_gpus else 0

        if selected_gpu in available_devices:
            torch.cuda.empty_cache()
            logging.info("🔥 LOCK AND LOAD! GPU ENGAGED! 🔥")
            device = torch.device(f"cuda:{selected_gpu}")
            torch.cuda.set_device(selected_gpu)
            torch.set_float32_matmul_precision("highest")
            logging.info(f"Using GPU: {selected_gpu} (cuda:{selected_gpu})")
        else:
            logging.info(f"⚠️ Warning: GPU {selected_gpu} is not available. Using CPU instead.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        logging.info("CUDA not available. Using CPU.")

    logging.info(f"Selected device: {device}")
    return device

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
def run():
    suppress_runtime_warnings()
    config = parse_args()
    date_time_dir = os.path.join(config["log_dir"], config["date"], config["time"])
    os.makedirs(date_time_dir, exist_ok=True)

    # checkpoints
    ckpts = get_checkpoints(config)

    # contex
    exp_ctx, target_features = get_contexts(config, date_time_dir)

    # attach prediction dataset
    dm = MeteoDatasetModule(
        data=None,
        ctx=exp_ctx,
        batch_size=config["batch_size"],
    )
    
    # fetch target future data
    kbh = Point(lat=55.6761, lon=12.5683)
    df_raw = get_hourly_example(kbh, start=datetime(2019, 6, 1), end=datetime(2019, 6, 3))
    dm.setup_prediction(df_raw)

    trainer = Trainer(accelerator="cpu")

    # create prediction output dir
    current_date = datetime.now().strftime("%Y%m%d")
    current_time = datetime.now().strftime("%H%M%S")
    out_dir = os.path.join(date_time_dir, f"{current_date}_{current_time}_predictions")
    os.makedirs(out_dir, exist_ok=True)

    # run prediction for each checkpoint
    for ckpt in ckpts:
        logging.info(f"running prediction using checkpoint: {ckpt}")
        model = MeteoVanillaTransformerEncoder.load_from_checkpoint(
            ckpt,
            model_ctx=exp_ctx.model,
            forecast_ctx=exp_ctx.forecast,
            input_features=dm._get_available_features(),
            target_features=target_features,
        )

        model.eval()

        preds_list = trainer.predict(model, datamodule=dm)
        preds = torch.cat(preds_list, dim=0).cpu().numpy()

        # save to filename based on ckpt
        base_name = os.path.basename(ckpt).replace(".ckpt", "")
        file_path = os.path.join(out_dir, f"forecast_{base_name}.npy")
        np.save(file_path, preds)

        print(f"✅ prediction complete for {base_name}, saved: {file_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
