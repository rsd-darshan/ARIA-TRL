"""
aria-trl: Continual Learning for LLM Fine-tuning

Brings ARIA's fast/slow pathway architecture to Hugging Face SFTTrainer,
enabling continual learning on sequential tasks without catastrophic forgetting.
"""

from .config import ARIAConfig
from .modules import PlasticityGatedMLP, TaskFastAdapter, MultiHeadScoreWrapper
from .trainer import ContinualSFTTrainer
from .consolidation import FisherConsolidator
from .baselines import EWCTrainer

__version__ = "1.1.0"
__author__ = "Darshan Poudel"

__all__ = [
    "ARIAConfig",
    "PlasticityGatedMLP",
    "TaskFastAdapter",
    "MultiHeadScoreWrapper",
    "ContinualSFTTrainer",
    "FisherConsolidator",
    "EWCTrainer",
]
