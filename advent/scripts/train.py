# --------------------------------------------------------
# AdvEnt training
# Copyright (c) 2019 valeo.ai
#
# Written by Tuan-Hung Vu
# --------------------------------------------------------
import argparse
import collections
import copy
import logging
import os
import os.path as osp
import pprint
import random
import shutil
import warnings

import numpy as np
import yaml
import torch
from sklearn import manifold
from torch.utils import data
from tqdm import tqdm

from advent.dataset.synthia import SYNTHIADataSet
from advent.model.deeplabv2 import get_deeplab_v2
from advent.dataset.gta5 import GTA5DataSet
from advent.dataset.cityscapes import CityscapesDataSet
from advent.domain_adaptation.config import cfg, cfg_from_file
from advent.domain_adaptation.train_UDA import train_domain_adaptation
from advent.model.hm import HybridMemory
import torch.nn.functional as F

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore")


def get_arguments():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Code for domain adaptation (DA) training")
    parser.add_argument('--cfg', type=str, default=None,
                        help='optional config file', )
    parser.add_argument("--random-train", action="store_true",
                        help="not fixing random seed.")
    parser.add_argument("--tensorboard", action="store_true",
                        help="visualize training loss with tensorboardX.")
    parser.add_argument("--viz-every-iter", type=int, default=None,
                        help="visualize results.")
    parser.add_argument("--exp-suffix", type=str, default=None,
                        help="optional experiment suffix")
    return parser.parse_args()


def main():
    # LOAD ARGS
    args = get_arguments()
    print('Called with args:')
    print(args)

    assert args.cfg is not None, 'Missing cfg file'
    cfg_from_file(args.cfg)
    # auto-generate exp name if not specified
    if cfg.EXP_NAME == '':
        cfg.EXP_NAME = f'{cfg.name}_{cfg.SOURCE}2{cfg.TARGET}_{cfg.TRAIN.MODEL}_{cfg.TRAIN.DA_METHOD}'

    if args.exp_suffix:
        cfg.EXP_NAME += f'_{args.exp_suffix}'
    # auto-generate snapshot path if not specified
    if cfg.TRAIN.SNAPSHOT_DIR == '':
        cfg.TRAIN.SNAPSHOT_DIR = osp.join(cfg.EXP_ROOT_SNAPSHOT, cfg.EXP_NAME)
        os.makedirs(cfg.TRAIN.SNAPSHOT_DIR, exist_ok=True)
    shutil.copytree("../../advent/", osp.join(cfg.TRAIN.SNAPSHOT_DIR, "advent"), dirs_exist_ok=True)
    device = cfg.GPU_ID

    # tensorboard
    if args.tensorboard:
        if cfg.TRAIN.TENSORBOARD_LOGDIR == '':
            cfg.TRAIN.TENSORBOARD_LOGDIR = osp.join(cfg.EXP_ROOT_LOGS, 'tensorboard', cfg.EXP_NAME)
        os.makedirs(cfg.TRAIN.TENSORBOARD_LOGDIR, exist_ok=True)
        if args.viz_every_iter is not None:
            cfg.TRAIN.TENSORBOARD_VIZRATE = args.viz_every_iter
    else:
        cfg.TRAIN.TENSORBOARD_LOGDIR = ''
    print('Using config:')
    pprint.pprint(cfg)

    # INIT
    _init_fn = None
    if not args.random_train:
        torch.manual_seed(cfg.TRAIN.RANDOM_SEED)
        torch.cuda.manual_seed(cfg.TRAIN.RANDOM_SEED)
        np.random.seed(cfg.TRAIN.RANDOM_SEED)
        random.seed(cfg.TRAIN.RANDOM_SEED)

        def _init_fn(worker_id):
            np.random.seed(cfg.TRAIN.RANDOM_SEED + worker_id)

    if os.environ.get('ADVENT_DRY_RUN', '0') == '1':
        return

    # LOAD SEGMENTATION NET
    assert osp.exists(cfg.TRAIN.RESTORE_FROM), f'Missing init model {cfg.TRAIN.RESTORE_FROM}'
    if cfg.TRAIN.MODEL == 'DeepLabv2':
        model = get_deeplab_v2(num_classes=cfg.NUM_CLASSES, multi_level=cfg.TRAIN.MULTI_LEVEL)
        saved_state_dict = torch.load(cfg.TRAIN.RESTORE_FROM)
        if 'DeepLab_resnet_pretrained_imagenet' in cfg.TRAIN.RESTORE_FROM:
            new_params = model.state_dict().copy()
            for i in saved_state_dict:
                i_parts = i.split('.')
                if not i_parts[1] == 'layer5':
                    new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
            model.load_state_dict(new_params)
        else:
            model.load_state_dict(saved_state_dict)
    else:
        raise NotImplementedError(f"Not yet supported {cfg.TRAIN.MODEL}")
    model.to(device)
    print('Model loaded')

    # DATALOADERS
    if "GTA5" in cfg.SOURCE:
        source_dataset = GTA5DataSet(root=cfg.DATA_DIRECTORY_SOURCE,
                                     list_path=cfg.DATA_LIST_SOURCE,
                                     set=cfg.TRAIN.SET_SOURCE,
                                     max_iters=cfg.TRAIN.MAX_ITERS * cfg.TRAIN.BATCH_SIZE_SOURCE,
                                     crop_size=cfg.TRAIN.INPUT_SIZE_SOURCE,
                                     mean=cfg.TRAIN.IMG_MEAN)
    else:
        source_dataset = SYNTHIADataSet(root=cfg.DATA_DIRECTORY_SOURCE_SYNTHIA,
                                        list_path=cfg.DATA_LIST_SOURCE_SYNTHIA,
                                        set=cfg.TRAIN.SET_SOURCE_SYNTHIA,
                                        max_iters=cfg.TRAIN.MAX_ITERS * cfg.TRAIN.BATCH_SIZE_SOURCE,
                                        crop_size=cfg.TRAIN.INPUT_SIZE_SOURCE,
                                        mean=cfg.TRAIN.IMG_MEAN)
        print("Loaded synthia dataset")

    source_loader = data.DataLoader(source_dataset,
                                    batch_size=cfg.TRAIN.BATCH_SIZE_SOURCE,
                                    num_workers=cfg.NUM_WORKERS,
                                    shuffle=True,
                                    pin_memory=True,
                                    worker_init_fn=_init_fn)

    target_dataset = CityscapesDataSet(root=cfg.DATA_DIRECTORY_TARGET,
                                       list_path=cfg.DATA_LIST_TARGET,
                                       set=cfg.TRAIN.SET_TARGET,
                                       info_path=cfg.TRAIN.INFO_TARGET,
                                       max_iters=cfg.TRAIN.MAX_ITERS * cfg.TRAIN.BATCH_SIZE_TARGET,
                                       crop_size=cfg.TRAIN.INPUT_SIZE_TARGET,
                                       mean=cfg.TRAIN.IMG_MEAN)
    target_loader = data.DataLoader(target_dataset,
                                    batch_size=cfg.TRAIN.BATCH_SIZE_TARGET,
                                    num_workers=cfg.NUM_WORKERS,
                                    shuffle=True,
                                    pin_memory=True,
                                    worker_init_fn=_init_fn)

    with open(osp.join(cfg.TRAIN.SNAPSHOT_DIR, 'train_cfg.yml'), 'w') as yaml_file:
        yaml.dump(cfg, yaml_file, default_flow_style=False)

    if cfg.TRAIN.switchcontra:
        # initialize HybridMemory
        # src_center = calculate_src_center(source_loader, device, model)
        # torch.save(src_center, "../../src_center_minent_all.pkl")
        src_center = torch.load("../../src_center_minent_all.pkl").to(device)

        tgt_center = calculate_tgt_center(target_loader, device, model, cfg.NUM_CLASSES, src_center, cfg)
        torch.save(tgt_center, "../../tgt_center_minent_all.pkl")
        # tgt_center = torch.load("../../tgt_center_minent_all.pkl").to(device)

        # Hybrid memory 存储源域的原型（需要每次迭代更新）和目标域的聚类后的原型，聚类时根据判别标准进行选择
        src_memory = HybridMemory(model.num_features, cfg.NUM_CLASSES,
                                  temp=cfg.TRAIN.contra_temp, momentum=cfg.TRAIN.contra_momentum, device=device).to(device)
        tgt_memory = HybridMemory(model.num_features, cfg.NUM_CLASSES,
                                  temp=cfg.TRAIN.contra_temp, momentum=cfg.TRAIN.contra_momentum, device=device).to(device)

        src_memory.features = src_center
        tgt_memory.features = tgt_center

    # UDA TRAINING
    train_domain_adaptation(model, source_loader, target_loader, cfg, src_memory, tgt_memory)


def calculate_src_center(source_all_dataloader, device, network):
    feat_dict = collections.defaultdict(list)
    with torch.no_grad():
        for i, (source_img, source_label, a, b) in tqdm(enumerate(source_all_dataloader)):
            source_img = source_img.to(device)
            # 获得经过 decoder 没有分类的特征
            class_base, class_high, feat_src = network(source_img)  # 4, 256, 160, 320
            source_label = F.interpolate(source_label.unsqueeze(1), feat_src.size()[2:], mode="nearest") \
            # 每一个特征根据当前方位的标签，归入到某个类别，并根据个数求平均 label (4, 256, 160, 320)，每个batch累计到最后平均
            class_high = F.interpolate(class_high, feat_src.size()[2:], mode="nearest")

            source_label_one = process_label(device, source_label.to(device))
            pred_label = process_label(device, F.softmax(class_high, dim=1).argmax(dim=1, keepdim=True).float())
            pred_correct = source_label_one * pred_label
            scale_factor = F.adaptive_avg_pool2d(pred_correct, output_size=1)

            for n in range(feat_src.size(0)):
                for t in range(19):
                    if scale_factor[n][t] == 0 or (pred_correct > 0).sum() < 1:
                        continue
                    s = feat_src[n] * pred_correct[n][t]
                    s = F.adaptive_avg_pool2d(s, output_size=1) / (scale_factor[n][t] + 1e-6)
                    # average pool 除以特征图大小求平均，每个类都一样，因此需要除以权重因子
                    feat_dict[t].append(s.unsqueeze(0).squeeze(2).squeeze(2))

            if i == 8000:
                break

        src_center = [torch.cat(feat_dict[cls], 0).mean(0, True) for cls in sorted(feat_dict.keys())]  # (19, 256)
        src_center = torch.cat(src_center, 0)
        src_center = F.normalize(src_center, dim=1)
        print(feat_dict[1][0].shape)
        assert src_center.size(0) == 19, "the shape of source center is incorrect {}, {}".format(
            src_center.size(), feat_dict.keys(), feat_dict[1][0].shape)
        # normailze will not interfere feature diversity, cause tgt_centers aren't lack

        return src_center


def calculate_tgt_center(target_train_dataloader, device, network, num_classes, src_center, config):
    print("==Extracting target center==")
    feat_dict = collections.defaultdict(list)
    with torch.no_grad():
        for i, (target_img, _, _, _) in tqdm(enumerate(target_train_dataloader)):
            target_img = target_img.to(device)
            # 获得经过 decoder 没有分类的特征
            _, _, output_target = network(target_img)  # 4, 256, 160, 320

            distance = torch.zeros((output_target.size(0), num_classes,
                                    output_target.size(2), output_target.size(3))).to(
                device)  # 4, 19, 160, 320
            # src_center 19, 256
            for n in range(output_target.size(0)):
                for t in range(num_classes):
                    distance[n][t] = torch.norm(
                        output_target[n] - src_center[t].reshape((src_center[t].size(0), 1, 1)).to(
                            device),
                        dim=0)  # 160, 320

            dis_min, dis_min_idx = distance.min(dim=1, keepdim=True)  # 4, 1, 160, 320
            distance_second = copy.deepcopy(distance)
            distance_second[distance_second == dis_min.expand_as(distance)] = 1000
            dis_sec_min, dis_sec_min_idx = distance_second.min(dim=1, keepdim=True)  # 1, nbr_tgt

            instmask = abs(dis_min - dis_sec_min) < config.TRAIN.cluster_threshold
            if config.TRAIN.ignore_instances:
                dis_min_idx[instmask] = 255

            pred_label = process_label(device, dis_min_idx.float())
            scale_factor = F.adaptive_avg_pool2d(pred_label, output_size=1)

            for n in range(pred_label.size(0)):
                for t in range(num_classes):
                    if scale_factor[n][t] == 0:
                        continue
                    s = output_target[n] * pred_label[n][t]  # 256, 160, 320
                    s = F.adaptive_avg_pool2d(s, output_size=1) / (scale_factor[n][t] + 1e-6)  # 256, 1, 1
                    # average pool 除以特征图大小求平均，每个类都一样，因此需要除以权重因子
                    feat_dict[t].append(s.unsqueeze(0).squeeze(2).squeeze(2))

            if i == 3000:
                break

        tgt_center = [torch.cat(feat_dict[cls], 0).mean(0, keepdim=True) for cls in sorted(feat_dict.keys())]
        tgt_center = F.normalize(torch.cat(tgt_center, dim=0), dim=1)
        assert tgt_center.size(0) == 19, "the shape of tgt_center is incorrect {}, {}".format(
            tgt_center.size(), feat_dict.keys())

        return tgt_center


def tsne(source_all_dataloader, device, network):
    from matplotlib import pyplot as plt

    with torch.no_grad():
        for i, (src_img, source_label, _, _) in tqdm(enumerate(source_all_dataloader)):
            src_img = src_img.to(device)
            output_source, _ = network(src_img)  # 4, 256, 160, 320
            source_label = F.interpolate(source_label.unsqueeze(1), (160, 320), mode="bilinear",
                                         align_corners=False).int()

            tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
            X_tsne = tsne.fit_transform(output_source.permute(0, 2, 3, 1).reshape(-1, 2048).cpu().numpy()[0:-1:100])
            # pickle.dump(X_tsne, open("MAX_batch1_tsne_batch{}.pkl".format(i), "wb"))
            # X_tsne = pickle.load(open("AE_batch1_tsne.pkl", "rb"))

            '''嵌入空间可视化'''
            x_min, x_max = X_tsne.min(0), X_tsne.max(0)
            X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
            source_label = source_label.reshape(-1, 1).numpy()[0:-1:100]
            plt.figure()
            for ii in range(X_norm.shape[0]):
                if source_label[ii][0] == -1:
                    continue
                else:
                    plt.text(X_norm[ii, 0], X_norm[ii, 1], str(source_label[ii][0]),
                             color=plt.cm.Set1(source_label[ii][0]),
                             fontdict={'weight': 'bold', 'size': 6})
            plt.savefig("MAX_reshape_tsne_batch{}.png".format(i))
            print("saved MAX_reshape_tsne_batch{}.png".format(i))


def process_label(device, label):
    """
    :desc: turn the label into one-hot format
    """
    batch, channel, w, h = label.size()
    pred1 = torch.zeros(batch, 20, w, h).to(device)
    # Return a tensor of elements selected from either :attr`x` or :attr:`y`,
    # depending on :attr:`condition
    label_trunk = torch.where(label<19, label, torch.Tensor([19]).to(device))
    #  place 1 on label place (replace figure > 19 with 19)
    pred1 = pred1.scatter_(1, label_trunk.long(), 1)
    return pred1


if __name__ == '__main__':
    main()
