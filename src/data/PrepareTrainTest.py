from src.data.DataReader import DataReader

from src.data.DataPrepareInterface import DataPrepareInterface


class PrepareTrainTest(DataPrepareInterface):
    def __init__(self, train_filename: str, test_filename: str, validation: bool, separator: float):

        self.validation = validation
        self.separator = separator

        # Read and split the train data
        train_data = DataReader.read(self.train_filename)
        if self.validation:
            split_index = int(len(train_data) * self.separator)
            split_index = self._find_next_period(train_data, split_index)
            self.train_data = train_data[:split_index]
            self.validation_data = train_data[split_index:]
        else:
            self.train_data = train_data
            self.validation_data = None

        # Read the test data
        self.test_data = DataReader.read(self.test_filename)


    def _find_next_period(self, data, index):
        while index < len(data) and data[index] != '.':
            index += 1
        return index

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def get_validation_data(self):
        return self.validation_data
