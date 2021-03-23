
import argparse
import os
import os.path as osp
import pprint
import warnings

import numpy
import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils import data
from tqdm import tqdm

from advent.dataset.cityscapes import CityscapesDataSet
from advent.dataset.gta5 import GTA5DataSet
from advent.domain_adaptation.Eval_update import decode_labels
from advent.domain_adaptation.config import cfg, cfg_from_file
from advent.model.deeplabv2 import get_deeplab_v2
from advent.utils.func import fast_hist, per_class_iu

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore")


def get_arguments():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Code for evaluation")
    parser.add_argument('--cfg', type=str, default=None,
                        help='optional config file', )
    return parser.parse_args()

def load_checkpoint_for_evaluation(model, checkpoint, device):
    saved_state_dict = torch.load(checkpoint)
    model.load_state_dict(saved_state_dict)
    model.eval()
    model.cuda(device)

def main(config_file, modelname, src_tag, modelstem):
    # LOAD ARGS
    assert config_file is not None, 'Missing cfg file'
    cfg_from_file(config_file)
    # auto-generate exp name if not specified
    cfg.EXP_NAME = modelstem
    # auto-generate snapshot path if not specified
    if cfg.TRAIN.TENSORBOARD_LOGDIR == '':
        cfg.TRAIN.TENSORBOARD_LOGDIR = osp.join(cfg.EXP_ROOT_LOGS, 'tensorboard', cfg.EXP_NAME)
        os.makedirs(cfg.TRAIN.TENSORBOARD_LOGDIR, exist_ok=True)
    device = cfg.GPU_ID


    print('Using config:')
    pprint.pprint(cfg)
    # load models
    models = []
    model = get_deeplab_v2(num_classes=cfg.NUM_CLASSES,
                           multi_level=cfg.TEST.MULTI_LEVEL[0], cfg=cfg).to(device)
    models.append(model)

    # dataloaders
    test_dataset = CityscapesDataSet(root=cfg.DATA_DIRECTORY_TARGET,
                                     list_path=cfg.DATA_LIST_TARGET,
                                     set=cfg.TEST.SET_TARGET,
                                     info_path=cfg.TEST.INFO_TARGET,
                                     crop_size=cfg.TEST.INPUT_SIZE_TARGET,
                                     mean=cfg.TEST.IMG_MEAN,
                                     labels_size=cfg.TEST.OUTPUT_SIZE_TARGET)
    test_loader = data.DataLoader(test_dataset,
                                  batch_size=cfg.TEST.BATCH_SIZE_TARGET,
                                  num_workers=cfg.NUM_WORKERS,
                                  shuffle=False,
                                  pin_memory=True)

    source_dataset = GTA5DataSet(root=cfg.DATA_DIRECTORY_SOURCE,
                                 list_path=cfg.DATA_LIST_SOURCE,
                                 set=cfg.TRAIN.SET_SOURCE,
                                 max_iters=cfg.TRAIN.MAX_ITERS * cfg.TRAIN.BATCH_SIZE_SOURCE,
                                 crop_size=cfg.TRAIN.INPUT_SIZE_SOURCE,
                                 mean=cfg.TRAIN.IMG_MEAN)

    source_loader = data.DataLoader(source_dataset,
                                    batch_size=cfg.TRAIN.BATCH_SIZE_SOURCE,
                                    num_workers=cfg.NUM_WORKERS,
                                    shuffle=False,
                                    pin_memory=True)

    # eval
    device = cfg.GPU_ID
    interp = nn.Upsample(size=(cfg.TEST.OUTPUT_SIZE_TARGET[1], cfg.TEST.OUTPUT_SIZE_TARGET[0]),
                         mode='bilinear', align_corners=True)
    writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)

    for model in models:
        load_checkpoint_for_evaluation(model, modelname, device)
    # eval
    if not src_tag:
        for index, batch in tqdm(enumerate(test_loader)):
            if index > 100:
                break
            hist = numpy.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
            image, label, _, name = batch
            with torch.no_grad():
                output = None
                for model in models:
                    pred_main = model(image.to(device))[1]
                    output = interp(pred_main).cpu().data[0].numpy()
                assert output is not None, 'Output is None'
                output = output.transpose(1, 2, 0)
                output = numpy.argmax(output, axis=2)
            label = label.numpy()[0]
            hist += fast_hist(label.flatten(), output.flatten(), cfg.NUM_CLASSES)

            labels_colors = decode_labels(numpy.expand_dims(label, axis=0), 2)
            preds_colors = decode_labels(numpy.expand_dims(output, axis=0), 2)
            inters_over_union_classes = per_class_iu(hist)
            mIOU = round(numpy.nanmean(inters_over_union_classes) * 100, 2)
            for index, (img, lab, color_pred) in enumerate(zip(image, labels_colors, preds_colors)):
                writer.add_image(name[0] + '/{}_Images'.format(str(mIOU)), img, index)
                writer.add_image(name[0] + '/{}_Labels'.format(str(mIOU)), lab, index)
                writer.add_image(name[0] + '/{}_preds'.format(str(mIOU)),  color_pred, index)

    else:
        for index, batch in tqdm(enumerate(source_loader)):
            if index > 100:
                break
            hist = numpy.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
            image, label, _, name = batch
            with torch.no_grad():
                output = None
                for model in models:
                    pred_main = model(image.to(device))[1]
                    output = interp(pred_main).cpu().data[0].numpy()
                assert output is not None, 'Output is None'
                output = output.transpose(1, 2, 0)
                output = numpy.argmax(output, axis=2)
            label = label.numpy()[0]
            hist += fast_hist(label.flatten(), output.flatten(), cfg.NUM_CLASSES)

            labels_colors = decode_labels(numpy.expand_dims(label, axis=0), 2)
            preds_colors = decode_labels(numpy.expand_dims(output, axis=0), 2)
            inters_over_union_classes = per_class_iu(hist)
            mIOU = round(numpy.nanmean(inters_over_union_classes) * 100, 2)
            for index, (img, lab, color_pred) in enumerate(zip(image, labels_colors, preds_colors)):
                writer.add_image(name[0] + '/{}_src_Images'.format(str(mIOU)),  img, index)
                writer.add_image(name[0] + '/{}_src_Labels'.format(str(mIOU)), lab, index)
                writer.add_image(name[0] + '/{}_src_preds'.format(str(mIOU)),  color_pred, index)


if __name__ == '__main__':
    args = get_arguments()
    models = [#"/root/tsy/ADVENT/experiments/snapshots/0222_ADAIN_alpha1_weights1_resize1024_bs4_GTA52Cityscapes_DeepLabv2_MinEnt/model_106000.pth",
              # "/root/tsy/ADVENT/experiments/snapshots/GTA2Cityscapes_DeepLabv2_MinEnt/model_94000.pth",
              # "/root/tsy/ADVENT/experiments/snapshots/0223_srconly_baseline_GTA52Cityscapes_DeepLabv2_MinEnt/model_94000.pth",
              # "/root/tsy/ADVENT/experiments/snapshots/0210_addcontra_clossw0.001_0.07temp_0.99momentum_moco_GTA52Cityscapes_DeepLabv2_MinEnt/model_24000.pth"]
              # "/root/tsy/ADVENT/experiments/snapshots/0222_ADAIN_alpha1_weights1_resize1024_bs4_GTA52Cityscapes_DeepLabv2_MinEnt/model_96000.pth",]
              # "/root/tsy/ADVENT/experiments/snapshots/GTA2Cityscapes_DeepLabv2_MinEnt/model_114000.pth",]
              # "/root/tsy/ADVENT/experiments/snapshots/0223_srconly_baseline_GTA52Cityscapes_DeepLabv2_MinEnt/model_94000.pth",]
              # "/root/tsy/ADVENT/experiments/snapshots/0210_addcontra_clossw0.001_0.07temp_0.99momentum_moco_GTA52Cityscapes_DeepLabv2_MinEnt/model_114000.pth"]
             "/root/tsy/ADVENT/experiments/snapshots/0227_ADAIN_cityscapes_addcontra_clossw0.001_0.07temp_0.99momentum_moco_rstrcitybest_realdst_GTA52Cityscapes_DeepLabv2_MinEnt/model_60000.pth",]
    stems = [# "PICS_0222_ADAIN_alpha1_weights1_resize1024_bs4_GTA52Cityscapes_DeepLabv2_MinEnt",
             # "PICS_GTA2Cityscapes_DeepLabv2_MinEnt",
             # 'PICS_0223_srconly_baseline_GTA52Cityscapes_DeepLabv2_MinEnt',
             # "PICS_0210_addcontra_clossw0.001_0.07temp_0.99momentum_moco_GTA52Cityscapes_DeepLabv2_MinEnt",
             # "PICS_0222_ADAIN_alpha1_weights1_resize1024_bs4_GTA52Cityscapes_DeepLabv2_MinEnt",]
             # "PICS_GTA2Cityscapes_DeepLabv2_MinEnt",]
             # "PICS_0223_srconly_baseline_GTA52Cityscapes_DeepLabv2_MinEnt",]
             # "PICS_0210_addcontra_clossw0.001_0.07temp_0.99momentum_moco_GTA52Cityscapes_DeepLabv2_MinEnt"]
            "PICS_0227_ADAIN_cityscapes_addcontra_clossw0.001_0.07temp_0.99momentum_moco_rstrcitybest_realdst_GTA52Cityscapes_DeepLabv2_MinEnt"]

    src_tag = False
    for i, modelname in enumerate(models):
        if i >= 3:
            src_tag = True
        print('Called with args:')
        print(args)
        main(args.cfg, modelname, src_tag, modelstem=stems[i])



