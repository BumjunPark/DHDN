import torch.utils.data as data
import torch
import h5py

class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        self.path = file_path

    def __getitem__(self, index):
        hf = h5py.File(self.path, 'r')
        self.data = hf.get("data")
        self.target = hf.get("label")
        return torch.from_numpy(self.data[index,:,:,:]).float(), torch.from_numpy(self.target[index,:,:,:]).float()

    def __len__(self):
        hf = h5py.File(self.path, 'r')
        temp = hf.get("data")
        return temp.shape[0]