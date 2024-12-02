from abc import ABC, abstractmethod


class ModelInterface(ABC):
    @abstractmethod
    def predict(self, data):
        pass

    @abstractmethod
    def train(self, data, target):
        pass
