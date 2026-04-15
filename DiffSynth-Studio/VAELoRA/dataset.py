from torch.utils.data import Dataset, DataLoader
from utils import collate_ab_lora_samples

class LoRACheckpointDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def make_ab_collate_fn(lora_patterns):
        def collate_fn(batch):
            return collate_ab_lora_samples(batch, lora_patterns)
        return collate_fn