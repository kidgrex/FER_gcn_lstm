import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class GCNFrameEncoder(nn.Module):
    def __init__(self, in_channels=2, out_channels=468):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.relu(x)
        return torch.mean(x, dim=0)  # (468,)


class GCN_LSTM_Model(nn.Module):
    def __init__(self, gcn_out=468, lstm_hidden=32, num_classes=6):
        super().__init__()

        self.gcn = GCNFrameEncoder(2, gcn_out)

        self.lstm = nn.LSTM(
            input_size=gcn_out,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(lstm_hidden, num_classes)

    def forward(self, batch_x, edge_index):
        seq_embeddings = []

        for seq in batch_x:      # per video
            frame_embs = []
            for frame in seq:   # per frame
                emb = self.gcn(frame, edge_index)
                frame_embs.append(emb)

            frame_embs = torch.stack(frame_embs)
            seq_embeddings.append(frame_embs)

        padded = nn.utils.rnn.pad_sequence(
            seq_embeddings, batch_first=True
        )

        _, (h, _) = self.lstm(padded)
        logits = self.fc(h[-1])
        return logits