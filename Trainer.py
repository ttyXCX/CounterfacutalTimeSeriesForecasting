import torch
from torch import nn

from PredictorModels import MLP
from SelectorModels import LogisticRegression
from InterpolationMethods import InterpolateOriginal


class Trainer:
    def __init__(self, selector, selector_learning_rate,
                       predictor, predictor_learning_rate,
                       interpolator):
        self.selector = selector
        self.selector_learning_rate = selector_learning_rate

        self.predictor = predictor
        self.predictor_learning_rate = predictor_learning_rate

        self.interpolator = interpolator

    @staticmethod
    def get_device(device_id=0):
        cuda_idx = 'cuda:%d' % device_id
        return torch.device(cuda_idx if torch.cuda.is_available() else 'cpu')

    def fit(self):
        # TODO
        print()


# TODO

# test function
if __name__ == "__main__":
    input_length = 96
    output_length = 12
    selector_learning_rate = 1e-3
    predictor_learning_rate = 1e-3
    data = torch.randn(100, 96)

    selector = LogisticRegression(input_length=input_length, output_length=output_length)
    predictor = MLP(input_length=input_length, output_length=output_length, hidden_size=[64, 32])
    interpolator = InterpolateOriginal()

    trainer = Trainer(selector=selector, selector_learning_rate=selector_learning_rate,
                      predictor=predictor, predictor_learning_rate=predictor_learning_rate,
                      interpolator=interpolator)
    device = trainer.get_device()
