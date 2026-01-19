import os
import numpy as np
import torch
from torch.utils.data import Dataset


def parse_filename(filename):
    """
    Format: {ACTOR}_{VTYPE}_{EMOTION}_{REP}.npz
    contoh: 1001_DFA_ANG_XX.npz
    """
    name = filename.replace(".npz", "")
    parts = name.split("_")
    actor = parts[0]
    emotion = parts[2]
    return actor, emotion


class GraphSequenceDataset(Dataset):
    def __init__(self, files, label_map):
        """
        files: list of absolute file paths
        """
        self.files = files
        self.label_map = label_map

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        data = np.load(file_path)

        landmarks = data["landmarks_norm"]      # (T, N, 2)
        edge_index = data["edge_index"]         # (E, 2)

        _, emotion = parse_filename(os.path.basename(file_path))
        label = self.label_map[emotion]

        return {
            "x": torch.tensor(landmarks, dtype=torch.float32),
            "edge_index": torch.tensor(edge_index.T, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long)
        }


def collate_fn(batch):
    xs = [item["x"] for item in batch]
    edge_index = batch[0]["edge_index"]
    labels = torch.stack([item["label"] for item in batch])
    return {"x": xs, "edge_index": edge_index, "label": labels}