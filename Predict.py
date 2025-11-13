#!/usr/bin/env python3
"""
Predict.py
----------
Load ALL checkpoints from a given run and run inference on future Meteostat data.
"""
import warnings
import pandas as pd
import logging
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

import argparse
import os
import re
import json
from datetime import datetime
import torch
from pytorch_lightning import Trainer
from meteostat import Point
from DataPipelineWorkShop import get_hourly_example, PreprocessingContext, ForecastContext, ModelContext, ExperimentContext, make_predict_loader
from VanillaTransformer import MeteoVanillaTransformerEncoder

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
def make_single_window_dataframe(location: Point, 
                                 start_time: datetime, 
                                 exp_ctx: ExperimentContext) -> pd.DataFrame:
    """
    Construct a minimal Meteostat dataframe covering exactly one forecasting window.

    Args:
        location (Point): Meteostat location object.
        start_time (datetime): End of the observation window (the forecast will start right after this).
        exp_ctx (ExperimentContext): Contains forecast.window and forecast.horizon.
    """
    fc = exp_ctx.forecast
    window_hours = fc.window
    horizon_hours = fc.horizon

    # Fetch data covering just enough history for one forecast
    history_start = start_time - pd.Timedelta(hours=window_hours)
    history_end   = start_time + pd.Timedelta(hours=horizon_hours)

    df = get_hourly_example(location, history_start, history_end)

    # Defensive timestamp conversion
    if isinstance(df.index, pd.PeriodIndex):
        df.index = df.index.to_timestamp()

    return df

# ===========================================================
def run():
    config = parse_args()
    device = lock_and_load(config)
    date_time_dir = os.path.join(config["log_dir"], config["date"], config["time"])
    os.makedirs(date_time_dir, exist_ok=True)

    # checkpoints
    ckpts = get_checkpoints(config)

    # contex
    exp_ctx, target_features = get_contexts(config, date_time_dir)

    # fetch target future data
    kbh = Point(lat=55.6761, lon=12.5683)
    start_time = datetime(2019, 6, 3, 0, 0)
    df_single = make_single_window_dataframe(kbh, start_time, exp_ctx)
    pred_dl = make_predict_loader(df_single, exp_ctx, batch_size=1)

    available_feature_list = pred_dl.dataset._get_available_features()
    logging.info(f"Available features for prediction: {available_feature_list}")

    if torch.cuda.is_available():
        trainer = Trainer(
            accelerator="gpu",
            devices=device,          # or [0] if you want to be explicit
            max_epochs=1
        )
    else:
        trainer = Trainer(
            accelerator="cpu",
            devices=1,          # CPUAccelerator requires an int > 0
            max_epochs=1
        )


    # create prediction output dir
    out_dir = os.path.join(date_time_dir, "predictions")
    os.makedirs(out_dir, exist_ok=True)

    # run prediction for each checkpoint
    for ckpt in ckpts:
        logging.info(f"running prediction using checkpoint: {ckpt}")
        model = MeteoVanillaTransformerEncoder.load_from_checkpoint(
            ckpt,
            model_ctx=exp_ctx.model,
            forecast_ctx=exp_ctx.forecast,
            input_features=available_feature_list,
            target_features=target_features
        )

        model.eval()

        preds = trainer.predict(model, dataloaders=pred_dl)
        pred = torch.cat(preds, dim=0).cpu()   # shape (1, H, D)
        pred = pred[0]                         # remove batch dimension → (H, D)

        # Identify base timestamp (last observed hour before forecast)
        horizon = exp_ctx.forecast.horizon
        base_time = df_single.index[exp_ctx.forecast.window - 1]
        future_times = [base_time + pd.Timedelta(hours=k + 1) for k in range(horizon)]

        # Construct output DataFrame
        df_out = pd.DataFrame(pred.tolist(), columns=target_features)
        df_out.insert(0, "time", future_times)

        # Save
        current_date = datetime.now().strftime("%Y%m%d")
        current_time = datetime.now().strftime("%H%M%S")

        ckpt_base = os.path.splitext(os.path.basename(ckpt))[0]

        # Convert pattern like 'epoch=19-val_loss=0.1025' → 'epoch-19-val_loss-0.1025'
        ckpt_tag = ckpt_base.replace("=", "-")

        # Remove any accidental double dashes (just in case)
        ckpt_tag = re.sub(r"-+", "-", ckpt_tag)

        # Build final CSV path
        file_path = os.path.join(out_dir, f"{current_date}_{current_time}_{ckpt_tag}.csv")

        df_out.to_csv(file_path, index=False)
        print(f"✅ prediction complete, saved: {file_path}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    suppress_runtime_warnings()
    logging.basicConfig(level=logging.INFO)
    run()
