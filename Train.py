#!/usr/bin/env python3
"""
train_meteo_transformer.py
--------------------------
Train a vanilla Transformer encoder for meteorological sequence-to-sequence forecasting.
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
from tabulate import tabulate
import time
import os
import sys
import torch
import json
from datetime import datetime
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
from meteostat import Point
from DataPipelineWorkShop import get_hourly_example, PreprocessingContext, ForecastContext, ModelContext, ExperimentContext,  make_dataloaders
from VanillaTransformer import MeteoVanillaTransformerEncoder

# ===========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train Transformer on Meteostat data")
    parser.add_argument("--gpu", nargs="+", default=[], help="List of GPU IDs to use")
    parser.add_argument("--window_size", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=2)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument(
        "--target_features",
        nargs="+",
        default=["temp", "rhum", "prcp", "wspd", "sin_wdir", "cos_wdir"],
        help="List of target feature names to predict"
    )
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--date", type=str, required=True, help="Execution date in YYYYMMDD format")
    parser.add_argument("--time", type=str, required=True, help="Execution time in HHMMSS format")
    parser.add_argument("--tracker", type=str, default="none",
        choices=["wandb", "mlflow", "none"],
        help="Experiment tracker backend to use."
    )

    return vars(parser.parse_args())

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
def setup_logging(base_log_dir, current_date, current_time, tracker):
    """Create timestamped logging directories and log files based on provided date and time."""
    date_time_dir = os.path.join(base_log_dir, current_date, current_time)
    os.makedirs(date_time_dir, exist_ok=True)
    # ./log/20251103/123456/

    training_logfile = os.path.join(date_time_dir, f"{current_date}_{current_time}_training.log")
    # ./log/20251103/123456/20251103_123456_training.log
    tracker_logfile = os.path.join(date_time_dir, f"{current_date}_{current_time}_{tracker}.log")
    # ./log/20251103/123456/20251103_123456_<tracker>.log

    logging.basicConfig(
        filename=training_logfile,
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    logging.info(f"📁 Training logs: {training_logfile}")
    logging.info(f"📁 {tracker} logs:   {tracker_logfile}")
    return date_time_dir, training_logfile, tracker_logfile

def log_training_parameters(config: dict):
    logging.info("=" * 80)
    logging.info("TRAINING CONFIGURATION / HYPERPARAMETERS")
    logging.info("=" * 80)
    for k,v in config.items():
        logging.info(f"{k:20s} = {v}")
    logging.info("=" * 80)


# ===========================================================
def write_benchmark_summary(start_time, end_time, trainer, config, log_file_path: str):
    """
    Print and append a benchmark summary table to the training log file.

    Args:
        start_time (float): time.time() at training start.
        end_time (float): time.time() at training end.
        trainer (pl.Trainer): Lightning trainer instance.
        config (dict): Training configuration dictionary.
        log_file_path (str): Full path to the training log file.
    """
    elapsed = end_time - start_time
    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))

    try:
        steps_per_epoch = len(trainer.datamodule.train_dataloader())
    except Exception:
        steps_per_epoch = "N/A"

    total_iters = getattr(trainer, "global_step", 0)
    iters_per_sec = total_iters / elapsed if elapsed > 0 else 0.0

    # Architectural / training parameters summary
    arch_params = {
        "d_model": config.get("d_model"),
        "n_heads": config.get("n_heads"),
        "d_ff": config.get("d_ff"),
        "num_layers": config.get("num_layers"),
        "dropout": config.get("dropout"),
        "window": config.get("window_size"),
        "horizon": config.get("horizon"),
        "batch_size": config.get("batch_size"),
        "epochs": config.get("epochs"),
    }

    meta_summary = [
        ["Started", datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S")],
        ["Ended", datetime.fromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S")],
        ["Elapsed (h:m:s)", elapsed_str],
        ["Global Steps", total_iters],
        ["Steps per Epoch", steps_per_epoch],
        ["Iterations/sec", f"{iters_per_sec:.2f}"],
    ]

    text_output = []
    text_output.append("\n" + "=" * 80)
    text_output.append("🏁 TRAINING BENCHMARK SUMMARY")
    text_output.append("=" * 80)
    text_output.append(tabulate(meta_summary, headers=["Metric", "Value"], tablefmt="fancy_grid"))
    text_output.append("\n⚙️ MODEL ARCHITECTURE PARAMETERS")
    text_output.append(tabulate(arch_params.items(), headers=["Parameter", "Value"], tablefmt="fancy_grid"))
    text_output.append("=" * 80 + "\n")

    summary_text = "\n".join(text_output)

    # Print to console
    logging.info(summary_text)

    # Append to log file
    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(summary_text)

# ===========================================================
def build_contexts(config, log_dir):
    pre_ctx = PreprocessingContext()
    fc_ctx = ForecastContext(
        window=config["window_size"],
        horizon=config["horizon"],
        val_ratio=config["val_ratio"],
        test_ratio=config["test_ratio"],
    )
    model_ctx=ModelContext(
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        d_ff=config["d_ff"],
        num_layers=config["num_layers"],
        dropout=config["dropout"]
    )
    exp_ctx = ExperimentContext(
        preprocessing=pre_ctx,
        forecast=fc_ctx,
        model=model_ctx
    )
    ctx_json_path = os.path.join(log_dir, f"{config["date"]}_{config["time"]}_context.json")
    with open(ctx_json_path, "w") as f:
        json.dump({
            "preprocessing": vars(pre_ctx),
            "forecast"     : vars(fc_ctx),
            "model"        : vars(model_ctx),
            "target_features": config["target_features"],
        }, f, indent=2)

    return fc_ctx, model_ctx, exp_ctx

# ===========================================================

def setup_and_get_wandb_logger(config, project_name, run_name):
    import wandb
    from pytorch_lightning.loggers import WandbLogger
    wandb.init(project=project_name, config=config, name=run_name)
    wandb_logger = WandbLogger(project=project_name, config=config)
    return wandb_logger

def setup_and_get_mlflow_logger(config, project_name, run_name):
    from pytorch_lightning.loggers import MLFlowLogger
    mlf_logger = MLFlowLogger(
        experiment_name=project_name,
        run_name=run_name,
        tracking_uri="http://127.0.0.1:5000"
    )
    mlf_logger.log_hyperparams(config)
    return mlf_logger

# ===========================================================
def run():
    suppress_runtime_warnings()
    config = parse_args()
    device = lock_and_load(config)
    current_date, current_time = config["date"], config["time"]
    log_dir, training_log, tracker_log = setup_logging(
        config["log_dir"], current_date, current_time, config["tracker"]
    )

    # === log the hyperparameters ===
    log_training_parameters(config)
    # === Experimental Tracker LOGGER ===
    project_name = f"[{current_date}] MeteoTransformer"
    run_name = f"{current_time}"

    if config["tracker"] == "wandb":
        logger = setup_and_get_wandb_logger(config, project_name, run_name)
    elif config["tracker"] == "mlflow":
        logger = setup_and_get_mlflow_logger(config, project_name, run_name)
    else:  # none
        logger = None

    # === Example data ===
    logging.info("🌍 Fetching Meteostat hourly data for Copenhagen...")
    kbh = Point(lat=55.6761, lon=12.5683)
    df_raw = get_hourly_example(kbh, start=datetime(2013, 1, 1), end=datetime(2018, 12, 31))
    
    # context objects
    fc_ctx, model_ctx, exp_ctx = build_contexts(config, log_dir)

    logging.info("📦 Building DataModule...")
    
    train_dl, val_dl, test_dl = make_dataloaders(df_raw, exp_ctx, batch_size=config["batch_size"], num_workers=2)
    available_feature_list = train_dl.dataset._get_available_features()
    logging.info(f"Available features: {available_feature_list}")

    logging.info("⚙️ Building model...")
    model = MeteoVanillaTransformerEncoder(
        model_ctx=model_ctx,
        forecast_ctx=fc_ctx,
        input_features=available_feature_list,
        target_features=config["target_features"],
    )

    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min", 
        save_last=True,
    )
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, mode="min")

    progressbar = TQDMProgressBar(refresh_rate=50)
    
    trainer = Trainer(
        max_epochs=config["epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=config["gpu"] if torch.cuda.is_available() and config["gpu"] else 1,
        callbacks=[checkpoint_callback, early_stopping, progressbar],
        default_root_dir=checkpoint_dir,
        log_every_n_steps=20,
        deterministic=True,
        logger=logger,
        precision="bf16-mixed" if torch.cuda.is_available() else 32,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
    )

    logging.info("🚀 Starting training...")
    start_time = time.time()
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    end_time = time.time()
    write_benchmark_summary(start_time, end_time, trainer, config, training_log)
    logging.info("✅ Training complete.")
    logging.info(f"📂 Checkpoints saved in: {checkpoint_dir}")
    logging.info(f"📝 Training log: {training_log}")
    logging.info(f"🧪 {config['tracker']} log: {tracker_log}")
    
    if config["tracker"] == "wandb":
        try:wandb.finish()
        except:pass
    elif config["tracker"] == "mlflow":
        from mlflow import end_run
        try: end_run()
        except: pass

# ===========================================================
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    suppress_runtime_warnings()
    run()
    time.sleep(0.5)
    sys.exit(0)

# nohup python Train.py --gpu 0 --epochs 20 > logs/$(date +"%Y%m%d_%H%M%S")_train.log 2>&1 &
# nohup mlflow ui --port 5000 > logs/$(date +"%Y%m%d_%H%M%S")_mlflow.log 2>&1 &

# nohup python Train.py --gpu 0 --epochs 20 &
# nohup mlflow ui --port 5000 > logs/mlflow_ui.log 2>&1 &

# terminal 1
# sh RunMLflow.sh 

# terminal 2
# sh RunTraining.sh