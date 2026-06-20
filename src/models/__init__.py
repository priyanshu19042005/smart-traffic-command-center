"""ML layer: training, prediction, evaluation and model registry."""
from .registry import ModelRegistry
from .train import train_all, train_task
from .predict import Predictor, predict_task
from .evaluate import evaluate_task

__all__ = ["ModelRegistry", "train_all", "train_task",
           "Predictor", "predict_task", "evaluate_task"]
