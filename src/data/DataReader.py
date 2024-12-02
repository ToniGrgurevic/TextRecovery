
class DataReader:
    @staticmethod
    def read(file_path: str):
        with open(file_path, 'r',encoding="utf-8") as file:
            data = file.readlines()
        return [line.strip() for line in data]