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
from DataPipelineWorkShop import get_hourly_example, PreprocessingContext, ForecastContext, ModelContext, ExperimentContext, make_predict_loader, make_single_window_dataframe, MeteoPreprocessor
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
    p.add_argument("--window_size", type=int, default=24)
    p.add_argument("--horizon", type=int, default=12)
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
    closer_type = CloserType.from_string(ctx_dict["closer_type"])
    target_features = ctx_dict["target_features"]
    logging.info(f"Target features: {target_features}")

    exp_ctx = ExperimentContext(
        preprocessing=pre_ctx,
        forecast=fc_ctx,
        model=model_ctx
    )
    return exp_ctx, target_features, closer_type

# ===========================================================
def build_model_signature(exp_ctx: ExperimentContext) -> str:
    """
    Build a short hyperparameter signature string from the context.
    LR is intentionally left blank.
    """
    mc = exp_ctx.model
    fc = exp_ctx.forecast

    signature = (
        f"b{mc.d_ff}_"
        f"d{mc.d_model}_"
        f"h{mc.n_heads}_"
        f"ls{mc.starter_num_layers}_"
        f"cs{mc.closer_num_layers}_"
        f"win{fc.window}_"
        f"ho{fc.horizon}_"
        f"dr{mc.dropout}_"
        f"lr"
    )
    return signature

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
def run():
    config = parse_args()
    is_lock_and_loaded = lock_and_load(config)
    date_time_dir = os.path.join(config["log_dir"], config["date"], config["time"])
    os.makedirs(date_time_dir, exist_ok=True)
    prediction_dir = os.path.join("prediction", config["date"], config["time"])
    os.makedirs(prediction_dir, exist_ok=True)
    
    # checkpoints
    ckpts = get_checkpoints(config)

    # contex
    exp_ctx, target_features, closer_type = get_contexts(config, date_time_dir)

    # fetch target future data
    kbh = Point(lat=55.6761, lon=12.5683)
    what_year = 2021
    start_times = {
        "vernal_equinox_formid" : datetime(what_year, 3, 21, 0, 0),
        "vernal_equinox_eftmid" : datetime(what_year, 3, 21, 12, 0),
        "sankthans_formid" : datetime(what_year, 6, 21, 0, 0),
        "sankthans_eftmid" : datetime(what_year, 6, 21, 12, 0),
        "hangeulnal_formid": datetime(what_year, 10, 9, 0, 0),
        "hangeulnal_eftmid": datetime(what_year, 10, 9, 12, 0),
    }
    
    for when_key, start_time in start_times.items():
        logging.info(f"Now predicting {when_key}......")
        df_single = make_single_window_dataframe(kbh, start_time, exp_ctx)

        trainer = Trainer(
            accelerator="gpu" if is_lock_and_loaded else "cpu",
            devices=config["gpu"] if is_lock_and_loaded and config["gpu"] else 1,
            max_epochs=1
        )
        
        # run prediction for each checkpoint
        prediction_csvs = {}
        for ckpt in ckpts:
            logging.info(f"running prediction using checkpoint: {ckpt}")

            # Get the feature schema used during training
            trained_feature_list = get_input_features_from_checkpoint(ckpt)

            logging.info(f"Trained features from checkpoint: {trained_feature_list}")

            # Build prediction dataset using the same feature schema
            pred_dl = make_predict_loader(
                df_single,
                exp_ctx,
                batch_size=1,
                fixed_features=trained_feature_list,
            )

            prediction_feature_list = pred_dl.dataset._get_available_features()
            logging.info(f"Prediction features after schema lock: {prediction_feature_list}")

            assert prediction_feature_list == trained_feature_list, (
                "Prediction dataset feature schema does not match checkpoint feature schema."
            )
            
            # model = MeteoVanillaTransformerEncoder.load_from_checkpoint(
            #     ckpt,
            #     model_ctx=exp_ctx.model,
            #     forecast_ctx=exp_ctx.forecast,
            #     input_features=trained_feature_list,
            #     closer_type=closer_type
            # )
            
            model = MeteoInformerHourglassTransformer.load_from_checkpoint(
                checkpoint_path=ckpt,
                model_ctx=exp_ctx.model,
                forecast_ctx=exp_ctx.forecast,
                input_features=trained_feature_list,
                closer_type=closer_type
            )

            model.eval()

            preds = trainer.predict(model, dataloaders=pred_dl)
            pred = torch.cat(preds, dim=0).cpu()   # shape (1, H, D)
            pred = pred[0]                         # remove batch dimension → (H, D)

            # Identify base timestamp (last observed hour before forecast)
            horizon = exp_ctx.forecast.horizon
            base_time = df_single.index[exp_ctx.forecast.window - 1]
            future_times = [base_time + pd.Timedelta(hours=k + 1) for k in range(horizon)]

            # Use the model's resolved output feature names
            resolved_target_features = model.get_target_features()

            # Convert model-space predictions to DataFrame
            df_pred = pd.DataFrame(
                pred.detach().cpu().float().numpy(),
                columns=resolved_target_features,
            )

            # Inverse-transform back to physical units
            preprocessor = MeteoPreprocessor(
                use_cyclic=exp_ctx.preprocessing.use_cyclic,
                categorical_mode=exp_ctx.preprocessing.categorical_mode,
            )

            df_out = preprocessor.inverse_transform(df_pred)

            # Add forecast timestamps
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
            start_time_str = start_time.strftime("%Y%m%d_%H%M")
            file_path = os.path.join(prediction_dir, 
                                     f"{current_date}_{current_time}_{ckpt_tag}_{start_time_str}({when_key}).csv")
            prediction_csvs[ckpt] = file_path

            df_out.to_csv(file_path, index=False)
            print(f"✅ prediction complete, saved: {file_path}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    suppress_runtime_warnings()
    logging.basicConfig(level=logging.INFO)
    run()
