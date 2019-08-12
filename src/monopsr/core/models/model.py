from abc import abstractmethod


class Model:

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def create_feed_dict(self):
        pass

    @abstractmethod
    def loss(self, output_dict, gt_dict):
        pass

    @abstractmethod
    def format_predictions(self, output_rep, predictions, sample_dict):
        pass

    @abstractmethod
    def save_predictions(self, sample_name, predictions, sample_dict, output_dirs):
        pass

    @abstractmethod
    def evaluate_predictions(self, prediction_dict, gt_dict):
        pass
