"""Re-export shared TSMM architecture for web package."""

import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.architecture import CBAM, ModelEMA, ResNet50Backbone, VideoTSMModel, build_model

__all__ = ["CBAM", "ModelEMA", "ResNet50Backbone", "VideoTSMModel", "build_model"]
