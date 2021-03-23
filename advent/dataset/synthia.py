import imageio
import numpy as np
from PIL import Image

from advent.dataset.base_dataset import BaseDataset

imageio.plugins.freeimage.download()  # cannot download this inside the cloud


class SYNTHIADataSet(BaseDataset):
    def __init__(self, root, list_path, set='all',
                 max_iters=None, crop_size=(321, 321), mean=(128, 128, 128)):
        super().__init__(root, list_path, set, max_iters, crop_size, None, mean)

        # map to cityscape's ids
        # self.id_to_trainid = {1: 10, 2: 2, 3: 0, 4: 1, 5: 4,
        #                       6: 8, 7: 5, 8: 13, 9: 7, 10: 11,
        #                       11: 18, 12: 17, 15: 6, 16: 9,
        #                       17: 12, 18: 14, 19: 15, 20: 16, 21: 3}

        self.synthia_set_16 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]

        self.id_to_trainid = {1: 10, 2: 2, 3: 0, 4: 1, 5: 4,
                              6: 8, 7: 5, 8: 13, 9: 7, 10: 11,
                              11: 18, 12: 17, 15: 6,
                              17: 12, 19: 15, 21: 3}

    def get_metadata(self, name):
        img_file = self.root / 'RGB' / name
        label_file = self.root / 'GT/LABELS' / name
        return img_file, label_file

    def get_labels(self, filename):
        gt_image = imageio.imread(filename, format='PNG-FI')[:, :, 0]
        gt_image = Image.fromarray(np.uint8(gt_image))
        img = gt_image.resize(self.labels_size, Image.NEAREST)
        img = np.asarray(img, np.float32)
        # print(img.shape)  # 640, 1280
        return img

    def __getitem__(self, index):
        img_file, label_file, name = self.files[index]
        image = self.get_image(img_file)
        label = self.get_labels(label_file)
        # re-assign labels to match the format of Cityscapes
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        for k, v in zip(self.synthia_set_16, range(16)):
            label_copy[label_copy == k] = v
        # print('label set, second map', np.unique(label_copy), len(np.unique(label_copy)), name)
        if len(np.unique(label_copy)) == 17:
            print(np.unique(label_copy), name)
        image = self.preprocess(image)
        return image.copy(), label_copy.copy(), np.array(image.shape), name
