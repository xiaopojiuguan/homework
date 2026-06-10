"""算法基类和工厂模式"""
from abc import ABC, abstractmethod
import numpy as np


class BaseClassifier(ABC):
    """所有分类器的抽象基类"""

    def __init__(self, name):
        self.name = name
        self.model = None
        self.is_trained = False
        self.train_losses = None
        self.val_losses = None

    @abstractmethod
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """训练模型"""
        ...

    @abstractmethod
    def predict(self, X):
        """预测类别"""
        ...

    @abstractmethod
    def predict_proba(self, X):
        """预测概率"""
        ...

    def get_train_losses(self):
        return self.train_losses

    def get_val_losses(self):
        return self.val_losses


class AlgorithmFactory:
    """算法工厂：注册和创建分类器"""

    _registry = {}

    @classmethod
    def register(cls, name, builder):
        cls._registry[name] = builder

    @classmethod
    def create(cls, name, **kwargs):
        if name not in cls._registry:
            raise ValueError(f"Unknown algorithm: {name}. Available: {list(cls._registry.keys())}")
        return cls._registry[name](**kwargs)

    @classmethod
    def list_algorithms(cls):
        return list(cls._registry.keys())
