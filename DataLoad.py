import torch
import itertools
import torch.utils.data as Data
import numpy as np
class Dataset_epoch(Data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, names, norm=False):
        'Initialization'
        self.names = names
        self.norm = norm
        self.index_pair = list(itertools.permutations(names, 2))

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.index_pair)

  def __getitem__(self, step):
        'Generates one sample of data'
        # Select sample
        img_A = np.load(self.index_pair[step][0])[np.newaxis]
        img_B = np.load(self.index_pair[step][1])[np.newaxis]



        return torch.from_numpy(img_A).float(), torch.from_numpy(img_B).float()