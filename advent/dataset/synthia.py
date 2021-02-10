import imageio
import numpy as np
from PIL import Image

from advent.dataset.base_dataset import BaseDataset


class SYNTHIADataSet(BaseDataset):
    def __init__(self, root, list_path, set='all',
                 max_iters=None, crop_size=(321, 321), mean=(128, 128, 128)):
        super().__init__(root, list_path, set, max_iters, crop_size, None, mean)

        # map to cityscape's ids
        self.id_to_trainid = {1: 10, 2: 2, 3: 0, 4: 1, 5: 4, 6: 8, 7: 5, 8: 13,
                              9: 7, 10: 11, 11: 18, 12: 17, 15: 6, 16: 9, 17: 12,
                              18: 14, 19: 15, 20: 16, 21: 3}

    def get_metadata(self, name):
        img_file = self.root / 'RGB' / name
        label_file = self.root / 'GT/LABELS' / name
        return img_file, label_file

    def get_labels(self, filename):
        gt_image = Image.open(filename)
        img = gt_image.resize(self.labels_size, Image.NEAREST)
        img = np.asarray(img, np.float32)
        # print(img.shape)  # 640, 1280, 3
        return img[:, :, 1]

    def __getitem__(self, index):
        img_file, label_file, name = self.files[index]
        image = self.get_image(img_file)
        label = self.get_labels(label_file)
        # re-assign labels to match the format of Cityscapes
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        # print(label_copy.min(), label_copy.max(), label_copy.shape)  # 640, 1280
        image = self.preprocess(image)
        return image.copy(), label_copy.copy(), np.array(image.shape), name
