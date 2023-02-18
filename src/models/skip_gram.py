from torch.nn import Module, Sequential, Embedding, Linear, LogSoftmax

class SkipGram(Module):

    def __init__(self, n_vocabulary: int, n_embedding: int = 256) -> None:

        super().__init__()

        self.model = Sequential(
            Embedding(n_vocabulary, n_embedding),
            Linear(n_embedding, n_vocabulary)
        )

    def forward(self, x):

        scores = self.model(x)
        log_soft_max = LogSoftmax(dim=1)
        log_ps = log_soft_max(scores)

        return log_ps