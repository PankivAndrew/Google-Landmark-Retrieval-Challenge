import os
from PIL import Image
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io, transform


class ParisDataset(Dataset):
    """Paris Landmark dataset."""
    def __init__(self, root_dir, transform=None):
        """
        Args:
            # csv_file (string): Path to the csv file with all data.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.filenames = os.listdir(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # img_name = os.path.join(self.root_dir,
        #                         self.data.iloc[idx, 0])
        # image = io.imread(img_name)
        # label = self.data.iloc[idx, 1]
        # sample = {'image': image, 'label': label, 'image_name': img_name}
        if type(idx) == int:
            filename = os.path.join(self.root_dir, self.filenames[idx])
        elif type(idx) == str:
            filename = os.path.join(self.root_dir, idx + ".jpg")

        image = Image.open(filename)

        if self.transform:
            sample = self.transform(image)

        this_idx = filename[filename.rfind("/") + 1:filename.rfind(".")]

        return this_idx, image

#
# def show_landmarks(image):
#     """Show image with landmarks"""
#     plt.imshow(image)

# csv_file='../data/csv/data.csv',


paris_dataset = ParisDataset(root_dir='../data/paris_dataset/')

print(paris_dataset[1000])
# fig = plt.figure()
# # print(paris_dataset[0])
#
# for i in range(len(paris_dataset)):
#     # if paris_dataset[i].get("label") == "general":
#     sample = paris_dataset[i]
#     label = paris_dataset[i].get("label")
#     print(sample.get("image_name"))
#     ax = plt.subplot(1, 4, i + 1)
#     plt.tight_layout()
#     ax.set_title('Label {}'.format(label))
#     ax.axis('off')
#     show_landmarks(sample.get("image"))
#
#     if i == 3:
#         plt.show()
#         break
#
#     # print(i, sample['image'].shape, sample['label'].shape)
#
