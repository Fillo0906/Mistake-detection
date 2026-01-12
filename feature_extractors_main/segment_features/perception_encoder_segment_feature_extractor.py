"""
PerceptionEncoder Feature Extractor for Segment Features

PerceptionEncoder is a foundation model for multimodal perception that can handle
video, audio, text and other modalities. This extractor extracts video embeddings
from video segments using PerceptionEncoder.

Reference: Rawat et al., "Towards Unique and Informative Captioning of Images"
"""

import argparse
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

# Configure logging
log_directory = os.path.join(os.getcwd(), 'logs')
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

log_file_path = os.path.join(log_directory, "perception_encoder_extraction.log")
logging.basicConfig(filename=log_file_path, filemode='a', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s')
logger = logging.getLogger(__name__)


class PerceptionEncoderVideoProcessor:
    """Processes videos and extracts features using PerceptionEncoder model"""
    
    def __init__(self, feature_extractor, video_transform):
        self.feature_extractor = feature_extractor
        self.video_transform = video_transform
        self.fps = 30
        self.num_frames_per_feature = 30

    def process_video(self, video_name, video_directory_path, output_features_path):
        """
        Process a single video and extract PerceptionEncoder features
        
        Args:
            video_name: Name of the video file
            video_directory_path: Path to video directory
            output_features_path: Path to save extracted features
        """
        segment_size = self.fps / self.num_frames_per_feature
        video_path = os.path.join(video_directory_path, 
                                  f"{video_name}.mp4" if "mp4" not in video_name else video_name)

        output_file_path = os.path.join(output_features_path, video_name)

        if os.path.exists(f"{output_file_path}_{int(segment_size)}s_{int(1)}s.npz"):
            logger.info(f"Skipping video: {video_name}")
            return

        os.makedirs(output_features_path, exist_ok=True)

        try:
            video = EncodedVideo.from_path(video_path)
            video_duration = video.duration - 0.0

            logger.info(f"Processing video: {video_name}, duration: {video_duration}s")
            segment_end = max(video_duration - segment_size + 1, 1)
            stride = 1

            video_features = []
            for start_time in tqdm(np.arange(0, segment_end, segment_size),
                                   desc=f"Processing video segments for {video_name}"):
                end_time = start_time + segment_size
                end_time = min(end_time, video_duration)

                if end_time - start_time < 0.04:
                    continue

                video_data = video.get_clip(start_sec=start_time, end_sec=end_time)
                segment_video_inputs = video_data["video"]

                segment_features = self._extract_features(segment_video_inputs)
                video_features.append(segment_features)

            video_features = np.vstack(video_features)
            np.savez(f"{output_file_path}_{int(segment_size)}s_{int(stride)}s.npz", video_features)
            logger.info(f"Saved features for video: {video_name}, shape: {video_features.shape}")
            
        except Exception as e:
            logger.error(f"Error processing video {video_name}: {str(e)}")
            raise

    def _extract_features(self, video_data_raw):
        """
        Extract features from video data
        
        Args:
            video_data_raw: Raw video tensor
            
        Returns:
            Feature numpy array
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        video_data_for_transform = {"video": video_data_raw, "audio": None}
        video_data = self.video_transform(video_data_for_transform)
        video_inputs = video_data["video"]
        
        # PerceptionEncoder expects single video input
        video_input = video_inputs.unsqueeze(0).to(device)
        
        with torch.no_grad():
            features = self.feature_extractor(video_input)
        
        return features.cpu().numpy()


def get_perception_encoder_video_transformation():
    """
    Get video transformation pipeline for PerceptionEncoder
    
    PerceptionEncoder uses:
    - 16 frames sampled uniformly
    - 256x256 resolution
    - ImageNet normalization
    """
    num_frames = 16
    side_size = 256
    crop_size = 256
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    video_transform = T.Compose(
        [
            UniformTemporalSubsample(num_frames),
            Lambda(lambda x: x / 255.0),
            NormalizeVideo(mean, std),
            ShortSideScale(size=side_size),
            CenterCropVideo(crop_size=(crop_size, crop_size)),
        ]
    )
    
    return ApplyTransformToKey(key="video", transform=video_transform)


def get_perception_encoder_feature_extractor(device="cuda"):
    """
    Load PerceptionEncoder model for feature extraction
    
    Returns the model without the classification head to get embeddings
    """
    try:
        # Try loading from huggingface transformers
        from transformers import AutoModel, AutoImageProcessor
        
        # Load the perception encoder model
        model = AutoModel.from_pretrained("facebook/perception-encoder-base")
        
    except Exception as e:
        logger.warning(f"Could not load PerceptionEncoder from transformers: {e}")
        logger.info("Attempting alternative loading method")
        
        try:
            # Alternative: try loading from torch hub
            model = torch.hub.load("facebookresearch/perception-encoder:main", "perception_encoder_base")
        except Exception as e2:
            logger.error(f"Failed to load PerceptionEncoder model: {e2}")
            raise RuntimeError(f"Cannot load PerceptionEncoder model. Please ensure the model is available. Error: {e2}")

    model = model.to(device)
    model = model.eval()
    return model


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="PerceptionEncoder Segment Feature Extraction")
    parser.add_argument("--video_path", type=str, default="/data/captain_cook/videos/gopro",
                        help="Path to videos directory")
    parser.add_argument("--output_path", type=str, default="/data/captain_cook/features/perception_encoder_segments/",
                        help="Path to save features")
    parser.add_argument("--num_threads", type=int, default=1,
                        help="Number of parallel threads")
    
    args = parser.parse_args()
    
    # Get model and transforms
    logger.info("Loading PerceptionEncoder model and transforms")
    video_transform = get_perception_encoder_video_transformation()
    feature_extractor = get_perception_encoder_feature_extractor()
    
    processor = PerceptionEncoderVideoProcessor(feature_extractor, video_transform)
    
    # Get list of videos
    video_files = [f for f in os.listdir(args.video_path) if f.endswith(".mp4")]
    logger.info(f"Found {len(video_files)} videos to process")
    
    # Process videos
    with concurrent.futures.ThreadPoolExecutor(args.num_threads) as executor:
        list(
            tqdm(
                executor.map(
                    lambda file: processor.process_video(file, args.video_path, args.output_path),
                    video_files
                ),
                total=len(video_files),
                desc="Processing videos"
            )
        )
    
    logger.info("Feature extraction complete")


if __name__ == "__main__":
    main()
