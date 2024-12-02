from abc import ABC, abstractmethod


class DataPrepareInterface(ABC):
    @abstractmethod
    def get_train_data(self):
        pass

    @abstractmethod
    def get_test_data(self):
        pass

    @abstractmethod
    def get_validation_data(self):
        pass