import torch
import torch.nn as nn


class ContributionRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()

        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size + hidden_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = torch.ones(self.hidden_size).unsqueeze(0)  - 1
        hn = torch.zeros(self.hidden_size).unsqueeze(0)

        for row in x:
            feed = torch.cat((row.unsqueeze(0), out), 1)
            self.out, hn = self.rnn(feed)

        return self.fc(out)