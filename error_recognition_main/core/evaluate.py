import argparse
from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import DataLoader

from base import fetch_model, test_er_model
from constants import Constants as const
from dataloader.CaptainCookStepDataset import CaptainCookStepDataset, collate_fn


@dataclass
class Config(object):
    backbone: str = "omnivore"
    modality: str = "video"
    phase: str = "train"
    segment_length: int = 1
    # Use this for 1 sec video features
    segment_features_directory: str = "/content/drive/MyDrive/PoliTo/Project/1s"

    ckpt_directory: str = "/content/drive/MyDrive/PoliTo/Project/error_recognition-main/checkpoints"
    split: str = "recordings"
    batch_size: int = 1
    test_batch_size: int = 1
    ckpt: Optional[str] = None
    seed: int = 1000
    device: str = "cuda"

    variant: str = const.TRANSFORMER_VARIANT
    #str = const.TRANSFORMER_VARIANT
    task_name: str = const.ERROR_RECOGNITION


def eval_er(config, threshold):
    model = fetch_model(config)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Load the model from the ckpt file
    model.load_state_dict(torch.load(config.ckpt_directory))
    model.eval()

    test_dataset = CaptainCookStepDataset(config, const.TEST, config.split)
    test_loader = DataLoader(test_dataset, batch_size=config.test_batch_size, collate_fn=collate_fn)

    # Calculate the evaluation metrics
    test_er_model(model, test_loader, criterion, config.device, phase="test", step_normalization=True, sub_step_normalization=True, threshold=threshold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, choices=[const.STEP_SPLIT, const.RECORDINGS_SPLIT], required=True)
    parser.add_argument("--backbone", type=str, choices=[const.SLOWFAST, const.OMNIVORE, const.PERCEPTION], required=True)
    parser.add_argument("--variant", type=str, choices=[const.MLP_VARIANT, const.TRANSFORMER_VARIANT, const.RNN_VARIANT], required=True)
    parser.add_argument("--phase", type=str, choices=[const.TEST], default=const.TEST)
    parser.add_argument("--modality", type=str, choices=[const.VIDEO])
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--threshold", type=float, required=True, default=0.5)
    parser.add_argument("--task_name", type=str, choices=[const.ERROR_RECOGNITION, const.ERROR_CATEGORY_RECOGNITION])
    parser.add_argument("--error_category", type=str, choices=[const.TECHNIQUE_ERROR, const.PREPARATION_ERROR,const.TEMPERATURE_ERROR, const.MEASUREMENT_ERROR, const.TIMING_ERROR])
    args = parser.parse_args()

    conf = Config()
    conf.split = args.split
    conf.backbone = args.backbone
    conf.variant = args.variant
    conf.phase = args.phase
    conf.modality = args.modality
    conf.ckpt_directory = args.ckpt
    conf.task_name = args.task_name
    # Only apply error_category when required
    if args.task_name == const.ERROR_CATEGORY_RECOGNITION:
        conf.error_category = args.error_category
    else:
        conf.error_category = None

    eval_er(conf, args.threshold)