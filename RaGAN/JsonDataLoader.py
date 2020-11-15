import json


class JsonDataLoader:
    def __init__(self, file_name):
        with open(file_name, "r") as data_file:
            self.data = json.load(data_file)

    def __getitem__(self, key):
        return self.data.__getitem__(key)

    def get_data(self):
        return self.data
