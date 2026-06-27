"""
aria-trl: Continual Learning for LLM Fine-tuning

Brings ARIA's fast/slow pathway architecture to Hugging Face SFTTrainer,
enabling continual learning on sequential tasks without catastrophic forgetting.
"""

from .config import ARIAConfig
from .modules import PlasticityGatedMLP, TaskFastAdapter
from .trainer import ContinualSFTTrainer
from .consolidation import FisherConsolidator

__version__ = "1.0.0"
__author__ = "Darshan Poudel"

__all__ = [
    "ARIAConfig",
    "PlasticityGatedMLP",
    "TaskFastAdapter",
    "ContinualSFTTrainer",
    "FisherConsolidator",
]
