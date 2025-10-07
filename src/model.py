import torch.nn as nn
from dataset import vocab

class LSTMModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 100)
        self.lstm = nn.LSTM(100, 150, batch_first=True)
        self.fc = nn.Linear(150, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        intermediate_hidden_state, (hidden_state, cell_state) = self.lstm(embedded)
        output = self.fc(hidden_state.squeeze(0))
        return output

model = LSTMModel(vocab_size=len(vocab))
