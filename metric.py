from torchmetrics import WordErrorRate
from data_utils import Word_process


class WER():
    def __init__(self, wordmodel: Word_process) -> None:
        self.wer = WordErrorRate()
        self.wordmodel = wordmodel

    def __call__(self, inputs, label):
        score = 0
        for i, j in zip(inputs.tolist(), label):
            score += self.wer(self.wordmodel.decode(i), j)
        return score / len(label)

        