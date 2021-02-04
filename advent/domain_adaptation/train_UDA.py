# --------------------------------------------------------
# Domain adpatation training
# Copyright (c) 2019 valeo.ai
#
# Written by Tuan-Hung Vu
# --------------------------------------------------------
import collections
import copy
import os
import sys
from pathlib import Path

import os.path as osp
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch import nn
from torchvision.utils import make_grid
from tqdm import tqdm

from advent.model.discriminator import get_fc_discriminator
from advent.utils.func import adjust_learning_rate, adjust_learning_rate_discriminator, adjust_threshold
from advent.utils.func import loss_calc, bce_loss
from advent.utils.loss import entropy_loss
from advent.utils.func import prob_2_entropy
from advent.utils.viz_segmask import colorize_mask


def train_advent(model, trainloader, targetloader, cfg):
    ''' UDA training with advent
    '''
    # Create the model and start the training.
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES
    viz_tensorboard = os.path.exists(cfg.TRAIN.TENSORBOARD_LOGDIR)
    if viz_tensorboard:
        writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)

    # SEGMNETATION NETWORK
    model.train()
    model.to(device)
    cudnn.benchmark = True
    cudnn.enabled = True

    # DISCRIMINATOR NETWORK
    # feature-level
    d_aux = get_fc_discriminator(num_classes=num_classes)
    d_aux.train()
    d_aux.to(device)

    # seg maps, i.e. output, level
    d_main = get_fc_discriminator(num_classes=num_classes)
    d_main.train()
    d_main.to(device)

    # OPTIMIZERS
    # segnet's optimizer
    optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                          lr=cfg.TRAIN.LEARNING_RATE,
                          momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # discriminators' optimizers
    optimizer_d_aux = optim.Adam(d_aux.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                 betas=(0.9, 0.99))
    optimizer_d_main = optim.Adam(d_main.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                  betas=(0.9, 0.99))

    # interpolate output segmaps
    interp = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear',
                         align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear',
                                align_corners=True)

    # labels for adversarial training
    source_label = 0
    target_label = 1
    trainloader_iter = enumerate(trainloader)
    targetloader_iter = enumerate(targetloader)
    for i_iter in tqdm(range(cfg.TRAIN.EARLY_STOP + 1)):

        # reset optimizers
        optimizer.zero_grad()
        optimizer_d_aux.zero_grad()
        optimizer_d_main.zero_grad()
        # adapt LR if needed
        adjust_learning_rate(optimizer, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_aux, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_main, i_iter, cfg)

        # UDA Training
        # only train segnet. Don't accumulate grads in disciminators
        for param in d_aux.parameters():
            param.requires_grad = False
        for param in d_main.parameters():
            param.requires_grad = False
        # train on source
        _, batch = trainloader_iter.__next__()
        images_source, labels, _, _ = batch
        pred_src_aux, pred_src_main = model(images_source.cuda(device))
        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux = interp(pred_src_aux)
            loss_seg_src_aux = loss_calc(pred_src_aux, labels, device)
        else:
            loss_seg_src_aux = 0
        pred_src_main = interp(pred_src_main)
        loss_seg_src_main = loss_calc(pred_src_main, labels, device)
        loss = (cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_AUX * loss_seg_src_aux)
        loss.backward()

        # adversarial training ot fool the discriminator
        _, batch = targetloader_iter.__next__()
        images, _, _, _ = batch
        pred_trg_aux, pred_trg_main = model(images.cuda(device))
        if cfg.TRAIN.MULTI_LEVEL:
            pred_trg_aux = interp_target(pred_trg_aux)
            d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_trg_aux)))
            loss_adv_trg_aux = bce_loss(d_out_aux, source_label)
        else:
            loss_adv_trg_aux = 0
        pred_trg_main = interp_target(pred_trg_main)
        d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main)))
        loss_adv_trg_main = bce_loss(d_out_main, source_label)
        loss = (cfg.TRAIN.LAMBDA_ADV_MAIN * loss_adv_trg_main
                + cfg.TRAIN.LAMBDA_ADV_AUX * loss_adv_trg_aux)
        loss = loss
        loss.backward()

        # Train discriminator networks
        # enable training mode on discriminator networks
        for param in d_aux.parameters():
            param.requires_grad = True
        for param in d_main.parameters():
            param.requires_grad = True
        # train with source
        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux = pred_src_aux.detach()
            d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_src_aux)))
            loss_d_aux = bce_loss(d_out_aux, source_label)
            loss_d_aux = loss_d_aux / 2
            loss_d_aux.backward()
        pred_src_main = pred_src_main.detach()
        d_out_main = d_main(prob_2_entropy(F.softmax(pred_src_main)))
        loss_d_main = bce_loss(d_out_main, source_label)
        loss_d_main = loss_d_main / 2
        loss_d_main.backward()

        # train with target
        if cfg.TRAIN.MULTI_LEVEL:
            pred_trg_aux = pred_trg_aux.detach()
            d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_trg_aux)))
            loss_d_aux = bce_loss(d_out_aux, target_label)
            loss_d_aux = loss_d_aux / 2
            loss_d_aux.backward()
        else:
            loss_d_aux = 0
        pred_trg_main = pred_trg_main.detach()
        d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main)))
        loss_d_main = bce_loss(d_out_main, target_label)
        loss_d_main = loss_d_main / 2
        loss_d_main.backward()

        optimizer.step()
        if cfg.TRAIN.MULTI_LEVEL:
            optimizer_d_aux.step()
        optimizer_d_main.step()

        current_losses = {'loss_seg_src_aux': loss_seg_src_aux,
                          'loss_seg_src_main': loss_seg_src_main,
                          'loss_adv_trg_aux': loss_adv_trg_aux,
                          'loss_adv_trg_main': loss_adv_trg_main,
                          'loss_d_aux': loss_d_aux,
                          'loss_d_main': loss_d_main}
        print_losses(current_losses, i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            torch.save(model.state_dict(), snapshot_dir / f'model_{i_iter}.pth')
            torch.save(d_aux.state_dict(), snapshot_dir / f'model_{i_iter}_D_aux.pth')
            torch.save(d_main.state_dict(), snapshot_dir / f'model_{i_iter}_D_main.pth')
            if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
                break
        sys.stdout.flush()

        # Visualize with tensorboard
        if viz_tensorboard:
            log_losses_tensorboard(writer, current_losses, i_iter)

            if i_iter % cfg.TRAIN.TENSORBOARD_VIZRATE == cfg.TRAIN.TENSORBOARD_VIZRATE - 1:
                draw_in_tensorboard(writer, images, i_iter, pred_trg_main, num_classes, 'T')
                draw_in_tensorboard(writer, images_source, i_iter, pred_src_main, num_classes, 'S')


def draw_in_tensorboard(writer, images, i_iter, pred_main, num_classes, type_):
    grid_image = make_grid(images[:3].clone().cpu().data, 3, normalize=True)
    writer.add_image(f'Image - {type_}', grid_image, i_iter)

    grid_image = make_grid(torch.from_numpy(np.array(colorize_mask(np.asarray(
        np.argmax(F.softmax(pred_main).cpu().data[0].numpy().transpose(1, 2, 0),
                  axis=2), dtype=np.uint8)).convert('RGB')).transpose(2, 0, 1)), 3,
                           normalize=False, range=(0, 255))
    writer.add_image(f'Prediction - {type_}', grid_image, i_iter)

    output_sm = F.softmax(pred_main).cpu().data[0].numpy().transpose(1, 2, 0)
    output_ent = np.sum(-np.multiply(output_sm, np.log2(output_sm)), axis=2,
                        keepdims=False)
    grid_image = make_grid(torch.from_numpy(output_ent), 3, normalize=True,
                           range=(0, np.log2(num_classes)))
    writer.add_image(f'Entropy - {type_}', grid_image, i_iter)


def train_minent(model, trainloader, targetloader, cfg, src_memory, tgt_memory):
    """ UDA training with minEnt
    """
    # Create the model and start the training.
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES
    viz_tensorboard = os.path.exists(cfg.TRAIN.TENSORBOARD_LOGDIR)
    if viz_tensorboard:
        writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)

    # SEGMNETATION NETWORK
    model.train()
    model.to(device)
    cudnn.benchmark = True
    cudnn.enabled = True

    # OPTIMIZERS
    # segnet's optimizer
    optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                          lr=cfg.TRAIN.LEARNING_RATE,
                          momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # interpolate output segmaps
    interp = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear',
                         align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear',
                                align_corners=True)

    trainloader_iter = enumerate(trainloader)
    targetloader_iter = enumerate(targetloader)
    for i_iter in tqdm(range(cfg.TRAIN.EARLY_STOP)):

        # reset optimizers
        optimizer.zero_grad()

        # adapt LR if needed
        adjust_learning_rate(optimizer, i_iter, cfg)

        # UDA Training
        # train on source
        _, batch = trainloader_iter.__next__()
        images_source, src_label, _, _ = batch
        pred_src_aux, pred_src_main, f_out_s = model(images_source.cuda(device))
        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux = interp(pred_src_aux)
            loss_seg_src_aux = loss_calc(pred_src_aux, src_label, device)
        else:
            loss_seg_src_aux = 0
        pred_src_main = interp(pred_src_main)
        loss_seg_src_main = loss_calc(pred_src_main, src_label, device)
        loss = (cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_AUX * loss_seg_src_aux)

        # adversarial training with minent
        _, batch = targetloader_iter.__next__()
        images, tgt_label, _, _ = batch
        pred_trg_aux, pred_trg_main, f_out_t = model(images.cuda(device))
        pred_trg_aux = interp_target(pred_trg_aux)
        pred_trg_main = interp_target(pred_trg_main)
        pred_prob_trg_aux = F.softmax(pred_trg_aux)
        pred_prob_trg_main = F.softmax(pred_trg_main)

        loss_target_entp_aux = entropy_loss(pred_prob_trg_aux)
        loss_target_entp_main = entropy_loss(pred_prob_trg_main)
        loss += (cfg.TRAIN.LAMBDA_ENT_AUX * loss_target_entp_aux
                 + cfg.TRAIN.LAMBDA_ENT_MAIN * loss_target_entp_main)

        # contrastive loss
        tgt_label = F.interpolate(tgt_label.float().unsqueeze(1), f_out_t.size()[2:],
                                  mode="nearest").int().view(-1, 1).to(device)  # 4,1, 160,320
        # t_labels_prob = torch.argmax(F.interpolate(f_out_t_class.data, f_out_t.size()[2:], mode="nearest"),
        #                         dim=1).flatten() + 19
        if cfg.TRAIN.adjthresholdpoly:
            threshold = adjust_threshold(cfg, i_iter)
        else:
            threshold = cfg.TRAIN.cluster_threshold
        t_labels = get_pseudo_labels(src_memory.features, f_out_t.data, num_classes, device, cfg, threshold).to(device)
        t_center = calculate_average(f_out_t, t_labels, device)
        t_labels[t_labels == 255] = -1
        tgt_label[tgt_label == 255] = -1

        src_label = F.interpolate(src_label.float().unsqueeze(1), size=f_out_s.size()[2:], mode="nearest").to(device)
        s_center = calculate_average(f_out_s, src_label, device)
        src_label[src_label == 255] = -1

        # print(s_center.shape, f_out_s.shape)  # [19, 2048] [1, 2048, 91, 161]
        # print(src_label.shape, src_label.min(), src_label.max()) 1, 1, 65, 129
        loss_s = src_memory(f_out_s.permute(0, 2, 3, 1).reshape(-1, 2048),
                            src_label.flatten().long(), torch.arange(19), s_center)
        loss_t = tgt_memory(f_out_t.permute(0, 2, 3, 1).reshape(-1, 2048),
                            t_labels.flatten().long(), torch.arange(19), t_center)

        loss += cfg.TRAIN.LAMBDA_CONTRA_S * loss_s + cfg.TRAIN.LAMBDA_CONTRA_T * loss_t

        loss.backward()
        optimizer.step()
        # 统计使用的标签中正确的部分
        t_labels = t_labels.flatten()
        a, b = t_labels[:-1:300], tgt_label[:-1:300].squeeze(1)
        a = a[b != -1]
        b = b[b != -1]
        current_pseudo_acc = 100 * (a == b).sum() / len(b)
        pseudo_acc = 100 * (t_labels[:-1:300] == (tgt_label[:-1:300].squeeze(1))).sum() / len(tgt_label[:-1:300])

        current_losses = {'loss_seg_src_aux': loss_seg_src_aux,
                          'loss_seg_src_main': loss_seg_src_main,
                          'loss_ent_aux': loss_target_entp_aux,
                          'loss_ent_main': loss_target_entp_main,
                          'loss_contra_src': loss_s,
                          'loss_contra_tgt': loss_t,
                          'pseudo_acc': pseudo_acc,
                          'current_pseudo_acc': current_pseudo_acc}

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            torch.save(model.state_dict(),
                       osp.join(cfg.TRAIN.SNAPSHOT_DIR, f'model_{i_iter}.pth'))
            if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
                break
        sys.stdout.flush()

        # Visualize with tensorboard
        if viz_tensorboard:
            log_losses_tensorboard(writer, current_losses, i_iter)

            if i_iter % cfg.TRAIN.TENSORBOARD_VIZRATE == cfg.TRAIN.TENSORBOARD_VIZRATE - 1:
                draw_in_tensorboard(writer, images, i_iter, pred_trg_main, num_classes, 'T')
                draw_in_tensorboard(writer, images_source, i_iter, pred_src_main, num_classes, 'S')
            if i_iter % cfg.TRAIN.print_lossrate == cfg.TRAIN.print_lossrate - 1:
                print_losses(current_losses, i_iter)


def calculate_average(features, label, device):  # 4,256, 160,320;  4,1, 160,320
    feat_dict = collections.defaultdict(list)
    # if 19 in label:
    #     start = 19
    # else:
    #     start = 0
    # pred_label = process_label((label - start).float())  # 4,20, 160,320
    pred_label = process_label(device, label.float())  # 4,20, 160,320
    scale_factor = F.adaptive_avg_pool2d(pred_label, output_size=1)  # 4,20, 1, 1
    for i in range(features.size(0)):
        for cls in range(19):
            tmp = features[i] * pred_label[i][cls]  # 256,160,320
            tmp = F.adaptive_avg_pool2d(tmp, output_size=1) / (scale_factor[i][cls] + 1e-6)  # 256 1 1
            feat_dict[cls].append(tmp.unsqueeze(0).squeeze(2).squeeze(2))

    tgt_center_pL = [torch.cat(feat_dict[cls], 0).mean(0, keepdim=True) for cls in sorted(feat_dict.keys())]
    tgt_center_pL = F.normalize(torch.cat(tgt_center_pL, dim=0), dim=1)

    return tgt_center_pL


def get_pseudo_labels(src_center, tgt_features, num_classes, device, config, threshold):  # 4,256,160,320 -> H,256
    distance = torch.zeros((tgt_features.size(0), num_classes,
                            tgt_features.size(2), tgt_features.size(3))).to(device)  # 4, 19, 160, 320
    # src_center 19, 256
    for n in range(tgt_features.size(0)):
        for t in range(num_classes):
            distance[n][t] = torch.norm(
                tgt_features[n] - src_center[t].reshape((src_center[t].size(0), 1, 1)).to(device),
                dim=0)  # 160, 320

    dis_min, dis_min_idx = distance.min(dim=1, keepdim=True)  # 4, 1, 160, 320
    distance_second = copy.deepcopy(distance)
    distance_second[distance_second == dis_min.expand_as(distance)] = 1000
    dis_sec_min, dis_sec_min_idx = distance_second.min(dim=1, keepdim=True)  # 1, nbr_tgt

    # if config["clusters"]["seperate"]:
    #     dis_min_idx += 19

    instmask = abs(dis_min - dis_sec_min) < threshold
    if config.TRAIN.ignore_instances:
        dis_min_idx[instmask] = 255
    else:
        outlier = 0
        for idx in torch.arange(dis_min.shape[1])[instmask.squeeze(0)]:
            dis_min_idx[0, idx] = num_classes + outlier
            outlier += 1

    return dis_min_idx  # 4, 1, 160,


def process_label(device, label):
    """
    :desc: turn the label into one-hot format
    """
    batch, channel, w, h = label.size()
    pred1 = torch.zeros(batch, 20, w, h).to(device)
    # Return a tensor of elements selected from either :attr`x` or :attr:`y`,
    # depending on :attr:`condition
    label_trunk = torch.where(19 > label, label, torch.Tensor([19]).to(device))
    #  place 1 on label place (replace figure > 19 with 19)
    pred1 = pred1.scatter_(1, label_trunk.long(), 1)
    return pred1


def print_losses(current_losses, i_iter):
    list_strings = []
    for loss_name, loss_value in current_losses.items():
        list_strings.append(f'{loss_name} = {to_numpy(loss_value):.3f} ')
    full_string = ' '.join(list_strings)
    tqdm.write(f'iter = {i_iter} {full_string}')


def log_losses_tensorboard(writer, current_losses, i_iter):
    for loss_name, loss_value in current_losses.items():
        writer.add_scalar(f'data/{loss_name}', to_numpy(loss_value), i_iter)


def to_numpy(tensor):
    if isinstance(tensor, (int, float)):
        return tensor
    else:
        return tensor.data.cpu().numpy()


def train_domain_adaptation(model, trainloader, targetloader, cfg, src_memory, tgt_memory):
    if cfg.TRAIN.DA_METHOD == 'MinEnt':
        train_minent(model, trainloader, targetloader, cfg, src_memory, tgt_memory)
    elif cfg.TRAIN.DA_METHOD == 'AdvEnt':
        train_advent(model, trainloader, targetloader, cfg)
    else:
        raise NotImplementedError(f"Not yet supported DA method {cfg.TRAIN.DA_METHOD}")
