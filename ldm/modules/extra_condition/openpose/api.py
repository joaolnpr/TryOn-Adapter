import numpy as np
import os
import torch.nn as nn

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import torch

from . import util
from .body import Body

remote_model_path = "https://huggingface.co/TencentARC/T2I-Adapter/resolve/main/third-party-models/body_pose_model.pth"


class OpenposeInference(nn.Module):

    def __init__(self):
        super().__init__()
        
        # Ensure models directory exists in TryOn-Adapter root
        # Find the TryOn-Adapter root directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        while current_dir and not current_dir.endswith('TryOn-Adapter'):
            parent = os.path.dirname(current_dir)
            if parent == current_dir:  # Reached filesystem root
                break
            current_dir = parent
        
        models_dir = os.path.join(current_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        body_modelpath = os.path.join(models_dir, "body_pose_model.pth")

        if not os.path.exists(body_modelpath):
            try:
                print(f"[OpenPose] Downloading model to {body_modelpath}...")
                from basicsr.utils.download_util import load_file_from_url
                load_file_from_url(remote_model_path, model_dir=models_dir)
                print(f"[OpenPose] ✅ Model downloaded successfully!")
            except Exception as e:
                print(f"[OpenPose] ❌ Failed to download model: {e}")
                raise Exception(f"OpenPose model download failed: {e}")
        
        try:
            self.body_estimation = Body(body_modelpath)
            print(f"[OpenPose] ✅ Model loaded successfully from {body_modelpath}")
        except Exception as e:
            print(f"[OpenPose] ❌ Failed to load model: {e}")
            raise Exception(f"OpenPose model loading failed: {e}")

    def forward(self, x):
        x = x[:, :, ::-1].copy()
        with torch.no_grad():
            candidate, subset = self.body_estimation(x)
            canvas = np.zeros_like(x)
            canvas = util.draw_bodypose(canvas, candidate, subset)
            canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        return canvas
