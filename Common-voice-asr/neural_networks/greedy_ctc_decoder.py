import torch
from typing import List
class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> List[str]:
        transcripts = []
        best_paths = torch.argmax(emission, dim=-1)

        for indices in best_paths:
            tokens = []
            prev = None
            for idx in indices:
                idx = idx.item()
                if idx != self.blank and idx != prev:
                    tokens.append(self.labels[idx])
                prev = idx

            joined = "".join(tokens).replace("|"," ").strip()
            words = joined.split()
            transcripts.append(words)
        return transcripts