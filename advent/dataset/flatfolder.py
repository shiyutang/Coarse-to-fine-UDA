from pathlib import Path

import numpy
from PIL import Image
from torch.utils import data
from torchvision import transforms


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, base_size, mean):
        super(FlatFolderDataset, self).__init__()
        self.paths = list(Path(root).glob('*'))
        self.mean = mean
        self.transform = self.adain_transform(base_size)

    def adain_transform(self, base_size):
        transform_list = [
            transforms.Resize(size=base_size),  # (h,w) 512, 1024
            transforms.ToTensor()
        ]
        return transforms.Compose(transform_list)

    def __getitem__(self, index):
        path = self.paths[index]
        img = numpy.array(Image.open(str(path)).convert('RGB'))
        img = img[:, :, ::-1]  # change to BGR
        img -= self.mean
        img = self.transform(Image.fromarray(img))
        return img

    def __len__(self):
        return len(self.paths)