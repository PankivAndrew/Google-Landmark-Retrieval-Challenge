import os

import torch
from torchvision import transforms

from scripts.vgg_baseline.vgg import Vgg16
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import Dataset

class GLRCDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.filenames = os.listdir(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if type(idx) == int:
            filename = os.path.join(self.root_dir, self.filenames[idx])
        elif type(idx) == str:
            filename = os.path.join(self.root_dir, idx + ".jpg")

        image = Image.open(filename)

        if self.transform:
            image = self.transform(image)

        this_idx = filename[filename.rfind("/") + 1:filename.rfind(".")]

        return this_idx, image

class VGGDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.filenames = os.listdir(root_dir)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if type(idx) == int:
            filename = os.path.join(self.root_dir, self.filenames[idx])
        elif type(idx) == str:
            filename = os.path.join(self.root_dir, idx)
        else:
            raise Exception("Invalid index")

        return torch.load(filename)


if __name__ == "__main__":
    dt = VGGDataset("../data/index_tensors")
    import time
    start = time.time()
    print(torch.sum((dt[0] - dt[1]) ** 2))
    end = time.time()
    print(end - start)