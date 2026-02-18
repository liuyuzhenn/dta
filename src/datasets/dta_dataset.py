from .base_dataset import BaseDataset
from torch.utils.data import Dataset
from scipy.io import loadmat
import torch
import os


class DtaDataset(BaseDataset):
    def __init__(self, configs):
        super().__init__(configs)

    def get_dataset(self, split):
        return _Dataset(split, self.configs)


class _Dataset(Dataset):
    def __init__(self, split, configs) -> None:
        super().__init__()
        self.configs = configs
        self.split = split
        self.path = configs['path']
        self.files = os.listdir(configs['path'])
        self.files = [f for f in self.files if f[-3:]=='mat']
        self.files.sort()
        train = configs['train_size']
        val = configs['val_size']
        test = configs['test_size']
        if self.split == 'train':
            self.files = self.files[:train]
        elif self.split == 'val':
            self.files = self.files[-val:]
        elif self.split == 'test':
            if val == 0 and train == 0:
                self.files = self.files
            else:
                self.files = self.files[-test-val:-val]
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file_path = os.path.join(self.path, self.files[index])
        data = loadmat(file_path)
        inds = torch.from_numpy(data['inds']).permute(1, 0).long()
        if torch.min(inds)==1:
            inds = inds-1

        t_gt = torch.from_numpy(data['t_gt']).float()
        t_rel = torch.from_numpy(data['rel_t']).float()
        t_rel /= torch.norm(t_rel, p=2, dim=-1, keepdim=True)
        data_ts = {
            't_gt': t_gt,
            't_rel': t_rel,
            'inds': inds
        }
        return data_ts
