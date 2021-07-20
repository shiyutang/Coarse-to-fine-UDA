# 建立文件夹为相对路径，相对当前所在路径
# shutil 最好也使用相对路径
import glob
import shutil
from pathlib import Path

if __name__ == '__main__':
    cities = ["aachen", "bochum", "bremen", "cologne", "darmstadt",
              "dusseldorf", "erfurt", "hamburg", "hanover", "jena", "krefeld",
              "monchengladbach", "strasbourg", "stuttgart", "tubingen", "ulm",
              "weimar", "zurich", "frankfurt", "lindau", "munster"]
    exp = "0311_alpha1_weights1_cityscapes_synthia_resize512crop256_bs4"
    root = "../Cityscapes/leftImg8bit"
    origin = "."

    Path(root + "/train").mkdir(parents=True, exist_ok=True)
    Path(root + "/val").mkdir(parents=True, exist_ok=True)
    for i, city in enumerate(cities):

        for n, file in enumerate(glob.glob(origin + '/*.png')):
            if file[2] in "0123456789":
                continue
            # print(file)
            if i < 18:
                Path(root + '/train/' + city).mkdir(exist_ok=True)
                if city in file:
                    shutil.move(file, root + '/train/' + city + '/')
                    print(file, "to", city)
            else:
                Path(root + '/val/' + city).mkdir(exist_ok=True)
                if city in file:
                    shutil.move(file, root + '/val/' + city + '/')
                    print(file, "to", city)