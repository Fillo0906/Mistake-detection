import argparse
import datetime
import glob
import os
import numpy as np
import torch
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)
import torchvision.transforms as T
import concurrent.futures
import logging
from tqdm import tqdm
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import NormalizeVideo, CenterCropVideo
from PIL import Image
from natsort import natsorted
import itertools
from perception_models_main.core.vision_encoder import pe

# Argument Parsing
def parse_arguments():
    parser = argparse.ArgumentParser(description="Script for processing methods.")
    parser.add_argument("--backbone", type=str, default="omnivore", help="Specify the method to be used.")
    return parser.parse_args()

# Video Processing
class VideoProcessor:
    def __init__(self, method, feature_extractor, video_transform):
        self.method = method
        self.feature_extractor = feature_extractor
        self.video_transform = video_transform

        self.fps = 30
        self.num_frames_per_feature = 30

    def process_video(self, video_name, video_directory_path, output_features_path):
        segment_size = self.fps / self.num_frames_per_feature
        video_path = os.path.join(video_directory_path, f"{video_name}.mp4" if "mp4" not in video_name else video_name)

        output_file_path = os.path.join(output_features_path, video_name)

        if os.path.exists(f"{output_file_path}_{int(segment_size)}s_{int(1)}s.npz"):
            logger.info(f"Skipping video: {video_name}")
            return

        os.makedirs(output_features_path, exist_ok=True)

        video = EncodedVideo.from_path(video_path)
        video_duration = video.duration - 0.0

        logger.info(f"video: {video_name} video_duration: {video_duration} s")
        segment_end = max(video_duration - segment_size + 1, 1)
        stride = 1

        video_features = []
        for start_time in tqdm(np.arange(0, segment_end, segment_size),
                               desc=f"Processing video segments for video {video_name}"):
            end_time = start_time + segment_size
            end_time = min(end_time, video_duration)

            if end_time - start_time < 0.04:
                continue

            video_data = video.get_clip(start_sec=start_time, end_sec=end_time)
            segment_video_inputs = video_data["video"]

            segment_features = extract_features(
                video_data_raw=segment_video_inputs,
                feature_extractor=self.feature_extractor,
                transforms_to_apply=self.video_transform,
                method=self.method
            )

            video_features.append(segment_features)

        video_features = np.vstack(video_features)
        np.savez(f"{output_file_path}_{int(segment_size)}s_{int(stride)}s.npz", video_features)
        logger.info(f"Finished extraction and saving video: {video_name} video_features: {video_features.shape}")

# Feature Extraction
def extract_features(video_data_raw, feature_extractor, transforms_to_apply, method):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    method = method.lower()
    
    if method == "perception_encoder":
        # Robustly find time/height/width/channel axes and extract a single RGB frame
        logger.info(f"Original video_data_raw - type: {type(video_data_raw)}, shape: {getattr(video_data_raw, 'shape', 'N/A')}")

        # Convert to numpy
        if isinstance(video_data_raw, torch.Tensor):
            video_np = video_data_raw.cpu().numpy()
        else:
            video_np = np.asarray(video_data_raw)

        logger.info(f"video_np.shape: {video_np.shape}, ndim: {video_np.ndim}, dtype: {video_np.dtype}")

        if video_np.ndim < 2 or video_np.ndim > 4:
            raise ValueError(f"Unsupported video array ndim: {video_np.ndim}")

        # Helper to pick axes representing height and width (two largest dims)
        shape = video_np.shape
        axes_sorted = sorted(range(len(shape)), key=lambda i: shape[i], reverse=True)
        # assume H and W are the two largest dims
        hw_axes = axes_sorted[:2]
        other_axes = [i for i in range(len(shape)) if i not in hw_axes]

        # Determine time axis: among other_axes pick the one with size >4 (likely time), else pick smallest
        time_axis = None
        channel_axis = None
        for ax in other_axes:
            if shape[ax] > 4:
                time_axis = ax
            else:
                channel_axis = ax
        if time_axis is None and len(other_axes) == 1:
            time_axis = other_axes[0]
        if channel_axis is None and len(other_axes) == 1 and shape[other_axes[0]] in [1, 3, 4]:
            channel_axis = other_axes[0]

        # If still unknown, use heuristics: channel likely 1/3/4
        if channel_axis is None:
            for i in range(len(shape)):
                if shape[i] in (1, 3, 4):
                    channel_axis = i
                    break

        # If time_axis still None, pick the smallest axis not channel
        if time_axis is None:
            candidates = [i for i in range(len(shape)) if i != channel_axis]
            time_axis = min(candidates, key=lambda i: shape[i])

        logger.info(f"Detected axes -> H/W axes: {hw_axes}, time_axis: {time_axis}, channel_axis: {channel_axis}")

        # Extract middle frame along time_axis if video has time dim
        if video_np.ndim == 4:
            middle_idx = shape[time_axis] // 2
            frame = np.take(video_np, indices=middle_idx, axis=time_axis)
        elif video_np.ndim == 3:
            # either [T,H,W] or [H,W,C] or [C,H,W]
            if time_axis is not None and shape[time_axis] > 4 and time_axis < 3:
                middle_idx = shape[time_axis] // 2
                frame = np.take(video_np, indices=middle_idx, axis=time_axis)
            else:
                frame = video_np
        else:
            # ndim == 2
            frame = video_np

        frame = np.asarray(frame)
        logger.info(f"Extracted frame shape (raw): {frame.shape}, ndim: {frame.ndim}")

        # Now ensure frame has shape [H, W, C]
        def find_valid_frame(arr):
            # Try without squeeze first to preserve single-channel dimension
            a = np.array(arr)
            
            # Handle 2D arrays (H, W) by adding channel dimension
            if a.ndim == 2:
                return np.expand_dims(a, axis=2)
            
            # Try squeeze but only specific dimensions (not the potential channel dim)
            a_squeezed = np.squeeze(a)
            
            # If squeeze resulted in 2D, expand dims to add channel
            if a_squeezed.ndim == 2:
                return np.expand_dims(a_squeezed, axis=2)
            
            # If already HWC and channels valid
            if a_squeezed.ndim == 3 and a_squeezed.shape[2] in (1, 2, 3, 4) and a_squeezed.shape[0] > 1 and a_squeezed.shape[1] > 1:
                return a_squeezed
            
            # Try permutations of axes to place channel last
            for perm in itertools.permutations(range(a_squeezed.ndim)):
                try:
                    cand = np.transpose(a_squeezed, perm)
                except Exception:
                    continue
                if cand.ndim == 3 and cand.shape[2] in (1, 2, 3, 4) and cand.shape[0] > 1 and cand.shape[1] > 1:
                    return cand
            
            # As fallback, if last dim is large but first two are 1, try reshape
            if a_squeezed.ndim == 3:
                h, w, c = a_squeezed.shape
                # if shape like (1,1,N) or (1,N,1) try to reshape to (h2,w2,3) if possible
                if min(h, w) == 1 and c > 4:
                    # try to interpret as (C,H,W) flattened incorrectly - attempt transpose
                    for perm in itertools.permutations(range(a_squeezed.ndim)):
                        cand = np.transpose(a_squeezed, perm)
                        if cand.ndim == 3 and cand.shape[2] in (1, 2, 3, 4) and cand.shape[0] > 1 and cand.shape[1] > 1:
                            return cand
            return None

        fixed = find_valid_frame(frame)
        if fixed is None:
            # Dump some debug info to assist
            logger.error(f"Cannot convert extracted frame to HWC usable by PIL. shape={frame.shape}, ndim={frame.ndim}, dtype={frame.dtype}")
            raise ValueError(f"Unexpected extracted frame ndim: {frame.ndim}, shape: {frame.shape}")
        frame = fixed
        # if single channel, convert to 3
        if frame.ndim == 3 and frame.shape[2] == 1:
            frame = np.repeat(frame, 3, axis=2)
        # if 2 channels, convert to 3 (e.g., optical flow or stereo)
        elif frame.ndim == 3 and frame.shape[2] == 2:
            # Duplicate one channel or pad with zeros
            frame = np.concatenate([frame, frame[:, :, :1]], axis=2)

        logger.info(f"Final frame shape before PIL: {frame.shape}")

        # Ensure dtype uint8 in 0-255
        if frame.dtype == np.float32 or frame.dtype == np.float64:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)

        frame_pil = Image.fromarray(frame).convert('RGB')
        frame_transform = T.Compose([
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        video_input = frame_transform(frame_pil).unsqueeze(0).to(device)
    else:
        video_data_for_transform = {"video": video_data_raw, "audio": None}
        video_data = transforms_to_apply(video_data_for_transform)
        video_inputs = video_data["video"]
        if method in ["omnivore"]:
            video_input = video_inputs[0][None, ...].to(device)
        elif method == "slowfast":
            video_input = [i.to(device)[None, ...] for i in video_inputs]
        elif method in ["x3d", "3dresnet", "egov"]:
            video_input = video_inputs.unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = feature_extractor(video_input)

    # Normalize outputs to a single tensor
    features = None
    if isinstance(outputs, torch.Tensor):
        features = outputs
    elif isinstance(outputs, tuple) or isinstance(outputs, list):
        # pick the first tensor-like element
        for o in outputs:
            if isinstance(o, torch.Tensor):
                features = o
                break
        if features is None:
            # try to find numpy arrays
            for o in outputs:
                if isinstance(o, np.ndarray):
                    features = torch.from_numpy(o)
                    break
    elif isinstance(outputs, dict):
        # common keys
        for k in ("image", "images", "features", "last_hidden_state", "embeddings"):
            if k in outputs:
                val = outputs[k]
                if isinstance(val, torch.Tensor):
                    features = val
                    break
                if isinstance(val, np.ndarray):
                    features = torch.from_numpy(val)
                    break
        if features is None:
            # pick first tensor in dict
            for val in outputs.values():
                if isinstance(val, torch.Tensor):
                    features = val
                    break
                if isinstance(val, np.ndarray):
                    features = torch.from_numpy(val)
                    break

    if features is None:
        raise ValueError(f"Unsupported model output type: {type(outputs)}")

    return features.cpu().numpy()

# Model Initialization
def get_video_transformation(name):
    name = name.lower()
    if name == "omnivore":
        num_frames = 32
        video_transform = T.Compose(
            [
                UniformTemporalSubsample(num_frames),
                T.Lambda(lambda x: x / 255.0),
                ShortSideScale(size=224),
                NormalizeVideo(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                TemporalCrop(frames_per_clip=32, stride=40),
                SpatialCrop(crop_size=224, num_crops=3),
            ]
        )
    elif name == "slowfast":
        slowfast_alpha = 4
        num_frames = 32
        side_size = 256
        crop_size = 256
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]

        class PackPathway(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, frames: torch.Tensor):
                fast_pathway = frames
                # Perform temporal sampling from the fast pathway.
                slow_pathway = torch.index_select(
                    frames,
                    1,
                    torch.linspace(
                        0, frames.shape[1] - 1, frames.shape[1] // slowfast_alpha
                    ).long(),
                )
                frame_list = [slow_pathway, fast_pathway]
                return frame_list

        video_transform = T.Compose(
            [
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(size=side_size),
                CenterCropVideo(crop_size),
                PackPathway(),
            ]
        )
    elif name == "x3d":
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        model_transform_params = {
            "x3d_xs": {
                "side_size": 182,
                "crop_size": 182,
                "num_frames": 4,
                "sampling_rate": 12,
            },
            "x3d_s": {
                "side_size": 182,
                "crop_size": 182,
                "num_frames": 13,
                "sampling_rate": 6,
            },
            "x3d_m": {
                "side_size": 256,
                "crop_size": 256,
                "num_frames": 16,
                "sampling_rate": 5,
            },
        }
        # Taking x3d_m as the model
        transform_params = model_transform_params["x3d_m"]
        video_transform = Compose(
            [
                UniformTemporalSubsample(transform_params["num_frames"]),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(size=transform_params["side_size"]),
                CenterCropVideo(
                    crop_size=(
                        transform_params["crop_size"],
                        transform_params["crop_size"],
                    )
                ),
            ]
        )
    elif name == "3dresnet":
        side_size = 256
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        crop_size = 256
        num_frames = 8
        video_transform = Compose(
            [
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(size=side_size),
                CenterCropVideo(crop_size=(crop_size, crop_size)),
            ]
        )
    elif name == "perception_encoder":
        num_frames = 8
        video_transform = Compose(
            [
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                ShortSideScale(size=224),
                CenterCropVideo(crop_size=(224, 224)),
                NormalizeVideo(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        raise ValueError(f"Unsupported backbone {name}")

    return ApplyTransformToKey(key="video", transform=video_transform)

def get_feature_extractor(name, device="cuda"):
    name = name.lower()
    if name == "omnivore":
        model_name = "omnivore_swinB_epic"
        model = torch.hub.load("facebookresearch/omnivore:main", model=model_name)
        model.heads = torch.nn.Identity()
    elif name == "slowfast":
        model = torch.hub.load("facebookresearch/pytorchvideo", "slowfast_r50", pretrained=True)
        model.heads = torch.nn.Identity()
    elif name == "x3d":
        model = torch.hub.load("facebookresearch/pytorchvideo", "x3d_m", pretrained=True)
        model.heads = torch.nn.Identity()
    elif name == "3dresnet":
        model = torch.hub.load("facebookresearch/pytorchvideo", "slow_r50", pretrained=True)
        model.heads = torch.nn.Identity()
    elif name == "perception_encoder":
        model_name = "PE-Core-B16-224"
        model = pe.CLIP.from_config(model_name, pretrained=True)
        model = model.to(device).eval()
        return model
    else:
        raise ValueError(f"Unsupported backbone {name}")

    feature_extractor = model
    feature_extractor = feature_extractor.to(device)
    feature_extractor = feature_extractor.eval()
    return feature_extractor

def main_hololens(is_sequential=False):
    hololens_directory_path = "/data/rohith/captain_cook/data/hololens/"
    output_features_path = f"/data/rohith/captain_cook/features/hololens/segments/{method}/"

    video_transform = get_video_transformation(method)
    feature_extractor = get_feature_extractor(method)

    processor = VideoProcessor(method, feature_extractor, video_transform)

    if not is_sequential:
        num_threads = 10
        with concurrent.futures.ThreadPoolExecutor(num_threads) as executor:
            for recording_id in os.listdir(hololens_directory_path):
                video_file_path = os.path.join(hololens_directory_path, recording_id, "sync", "pv")
                executor.submit(processor.process_video, recording_id, video_file_path, output_features_path)
    else:
        for recording_id in os.listdir(hololens_directory_path):
            video_file_path = os.path.join(hololens_directory_path, recording_id, "sync", "pv")
            processor.process_video(recording_id, video_file_path, output_features_path)

# Main
def main():
    video_files_path = "/content/drive/MyDrive/PoliTo/Project/GoProdownloaded/captain_cook_4d/gopro/Gopro_unzipped"
    output_features_path = f"/content/drive/MyDrive/Mistake_Detection/features/video/{method}"

    video_transform = get_video_transformation(method)
    feature_extractor = get_feature_extractor(method)

    processor = VideoProcessor(method, feature_extractor, video_transform)

    mp4_files = [file for file in os.listdir(video_files_path) if file.endswith(".mp4")]

    num_threads = 1
    with concurrent.futures.ThreadPoolExecutor(num_threads) as executor:
        list(
            tqdm(
                executor.map(
                    lambda file: processor.process_video(file, video_files_path, output_features_path), mp4_files
                ), total=len(mp4_files)
            )
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for processing methods.")
    parser.add_argument("--backbone", type=str, default="omnivore", help="Specify the method to be used.")
    args = parse_arguments()
    method = args.backbone.lower()

    log_directory = os.path.join(os.getcwd(), 'logs')
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    log_file_path = os.path.join(log_directory, f"{method}.log")
    logging.basicConfig(filename=log_file_path, filemode='a', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s')

    logger = logging.getLogger(__name__)

    # main_hololens(is_sequential=False)
    main()
