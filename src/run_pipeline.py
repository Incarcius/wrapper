"""
Entry point: streams frames from the Mid-70 and yields ML-ready (1024,3) arrays.

Usage:
    python src/run_pipeline.py --config config/pipeline_config.yaml
"""
import argparse
import logging
import time

import yaml
import numpy as np

from pipeline import PointCloudPipeline
from livox_interface import LivoxInterface

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("run_pipeline")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/pipeline_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    pipeline = PointCloudPipeline(config)
    ingest = LivoxInterface(config["ingestion"])
    ingest.start()

    logger.info("Pipeline running. Ctrl+C to stop.")
    try:
        while True:
            frame = ingest.get_latest_frame(timeout=1.0)
            if frame is None:
                continue
            raw_points, ts = frame
            processed, timings = pipeline.process(raw_points)
            
            # --- DGCNN ML Integration ---
            # DGCNN expects input shape: (Batch, Channels, NumPoints) -> (1, 3, 1024)
            if isinstance(processed, np.ndarray):
                dgcnn_input = np.expand_dims(processed.T, axis=0) 
            else:
                # Assuming PyTorch tensor
                dgcnn_input = processed.T.unsqueeze(0)
                
            # TODO: Pass dgcnn_input to your loaded DGCNN model here
            # logits = dgcnn_model(dgcnn_input)
            # predicted_class = logits.argmax(-1)
            # logger.info(f"Classified as: {predicted_class}")

        pass
    finally:
        ingest.stop()


if __name__ == "__main__":
    main()
