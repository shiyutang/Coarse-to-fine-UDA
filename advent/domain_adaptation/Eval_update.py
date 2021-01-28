# --------------------------------------------------------
# Domain adpatation evaluation
# Copyright (c) 2019 valeo.ai
#
# Written by Tuan-Hung Vu
# --------------------------------------------------------
import json
import logging
import os
import os.path as osp
import time

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
from tqdm import tqdm

from advent.dataset.cityscapes import DEFAULT_INFO_PATH
from advent.utils.eval import Eval
from advent.utils.func import per_class_iu, fast_hist
from advent.utils.serialization import pickle_dump, pickle_load


def evaluate_domain_adaptation(models, test_loader, source_loader, cfg,
                               fixed_test_size=True,
                               verbose=True):
    device = cfg.GPU_ID
    interp = None
    if fixed_test_size:
        interp = nn.Upsample(size=(cfg.TEST.OUTPUT_SIZE_TARGET[1], cfg.TEST.OUTPUT_SIZE_TARGET[0]), mode='bilinear',
                             align_corners=True)
    # eval
    if cfg.TEST.MODE == 'single':
        eval_single(cfg, models,
                    device, test_loader, interp, fixed_test_size,
                    verbose)
    elif cfg.TEST.MODE == 'best':
        eval_best(cfg, models,
                  device, test_loader, source_loader, interp, fixed_test_size,
                  verbose)
    else:
        raise NotImplementedError(f"Not yet supported test mode {cfg.TEST.MODE}")


def eval_single(cfg, models,
                device, test_loader, interp,
                fixed_test_size, verbose):
    assert len(cfg.TEST.RESTORE_FROM) == len(models), 'Number of models are not matched'
    for checkpoint, model in zip(cfg.TEST.RESTORE_FROM, models):
        load_checkpoint_for_evaluation(model, checkpoint, device)
    # eval
    hist = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
    for index, batch in tqdm(enumerate(test_loader)):
        image, label, _, name = batch
        if not fixed_test_size:
            interp = nn.Upsample(size=(label.shape[1], label.shape[2]), mode='bilinear', align_corners=True)
        with torch.no_grad():
            output = None
            for model, model_weight in zip(models, cfg.TEST.MODEL_WEIGHT):
                pred_main = model(image.to(device))[1]
                output_ = interp(pred_main).cpu().data[0].numpy()
                if output is None:
                    output = model_weight * output_
                else:
                    output += model_weight * output_
            assert output is not None, 'Output is None'
            output = output.transpose(1, 2, 0)
            output = np.argmax(output, axis=2)
        label = label.numpy()[0]
        hist += fast_hist(label.flatten(), output.flatten(), cfg.NUM_CLASSES)
    inters_over_union_classes = per_class_iu(hist)
    print(f'mIoU = \t{round(np.nanmean(inters_over_union_classes) * 100, 2)}')
    if verbose:
        display_stats(cfg, test_loader.dataset.class_names, inters_over_union_classes)


def eval_best(cfg, models, device, test_loader, source_loader, interp, fixed_test_size, verbose):
    assert len(models) == 1, 'Not yet supported multi models in this mode'
    assert osp.exists(cfg.TEST.SNAPSHOT_DIR[0]), 'SNAPSHOT_DIR is not found'
    start_iter = cfg.TEST.SNAPSHOT_STEP
    step = cfg.TEST.SNAPSHOT_STEP
    max_iter = cfg.TEST.SNAPSHOT_MAXITER
    cache_path = osp.join(cfg.TEST.SNAPSHOT_DIR[0], 'all_res.pkl')

    if osp.exists(cache_path):
        all_res = pickle_load(cache_path)
    else:
        all_res = {}

    cur_best_miou = -1
    cur_best_model = ''
    eval = Eval(cfg.NUM_CLASSES)
    logger = init_logger(cfg)
    writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)

    for i_iter in range(start_iter, max_iter + 1, step):
        restore_from = osp.join(cfg.TEST.SNAPSHOT_DIR[0], f'model_{i_iter}.pth')
        logger.info("restore_from"+ str(restore_from))
        if not osp.exists(restore_from):
            # continue
            if cfg.TEST.WAIT_MODEL:
                print('Waiting for model..!')
                while not osp.exists(restore_from):
                    time.sleep(5)

        logger.info("Evaluating model in target & source")
        if i_iter not in all_res.keys():
            load_checkpoint_for_evaluation(models[0], restore_from, device)
            # eval
            hist = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))

            # validate_target
            test_iter = iter(test_loader)
            eval.reset()
            for _ in tqdm(range(len(test_iter))):
                image, label, _, name = next(test_iter)  # 1, 3, 512, 1024  tensor
                if not fixed_test_size:
                    interp = nn.Upsample(size=(label.shape[1], label.shape[2]), mode='bilinear', align_corners=True)
                with torch.no_grad():
                    pred_main = models[0](image.cuda(device))[1]
                    output = interp(pred_main).cpu().data[0].numpy()  # 19, 1024, 2048
                    output = output.transpose(1, 2, 0)  # 
                    output = np.argmax(output, axis=2)

                label = label.numpy()[0]
                hist += fast_hist(label.flatten(), output.flatten(), cfg.NUM_CLASSES)  # label is 1024, 2048
                eval.add_batch(label, output)
            else:
                epoch = i_iter // step
                labels_colors = decode_labels(np.expand_dims(label, axis=0), 2)
                preds_colors = decode_labels(np.expand_dims(output, axis=0), 2)
                for index, (img, lab, color_pred) in enumerate(zip(image, labels_colors, preds_colors)):
                    writer.add_image(str(epoch) + '/Images', img, epoch)
                    writer.add_image(str(epoch) + '/Labels', lab, epoch)
                    writer.add_image(str(epoch) + '/preds', color_pred, epoch)

                def val_info(eval, name):
                    PA = eval.Pixel_Accuracy()
                    MPA = eval.Mean_Pixel_Accuracy()
                    MIoU = eval.Mean_Intersection_over_Union()
                    FWIoU = eval.Frequency_Weighted_Intersection_over_Union()
                    PC = eval.Mean_Precision()
                    logger.info("########## Eval{} ############".format(name))

                    logger.info('\nEpoch:{:.3f}, {} PA1:{:.3f}, MPA1:{:.3f}, MIoU1:{:.3f}, FWIoU1:{:.3f}, PC:{:.3f}'.
                                format(epoch, name, PA, MPA, MIoU, FWIoU, PC))
                    eval.Print_Every_class_Eval(logger=logger)

                    writer.add_scalar('PA' + name, PA, epoch)
                    writer.add_scalar('MPA' + name, MPA, epoch)
                    writer.add_scalar('MIoU' + name, MIoU, epoch)
                    writer.add_scalar('FWIoU' + name, FWIoU, epoch)
                    return PA, MPA, MIoU, FWIoU

                PA, MPA, MIoU, FWIoU = val_info(eval, "target")
                logger.info('\tCurrent target PA: {}, MPA: {}, MIoU: {}, FWIoU: {},'.format(PA, MPA, MIoU, FWIoU))


        # validate_source
            eval.reset()
            source_iter = iter(source_loader)
            for _ in tqdm(range(500)):
                image, label, _, name = next(source_iter)  # 3, 1024, 512 np array
                resize = nn.Upsample(size=(label.shape[1], label.shape[2]), mode='bilinear', align_corners=True)
                with torch.no_grad():
                    pred_main = models[0](image.cuda(device))[1]
                    output = resize(pred_main).cpu().data[0].numpy()  # 19, 2048, 1024
                    output = output.transpose(1, 2, 0)  #
                    output = np.argmax(output, axis=2)

                label = label.numpy()[0]
                eval.add_batch(label, output)
            else:
                epoch = i_iter // step
                labels_colors = decode_labels(np.expand_dims(label, axis=0), 2)
                preds_colors = decode_labels(np.expand_dims(output, axis=0), 2)
                for index, (img, lab, color_pred) in enumerate(zip(image, labels_colors, preds_colors)):
                    writer.add_image(str(epoch) + '/src_Images', img, epoch)
                    writer.add_image(str(epoch) + '/src_Labels', lab, epoch)
                    writer.add_image(str(epoch) + '/src_preds', color_pred, epoch)

                def val_info(Eval, name):
                    PA = Eval.Pixel_Accuracy()
                    MPA = Eval.Mean_Pixel_Accuracy()
                    MIoU = Eval.Mean_Intersection_over_Union()
                    FWIoU = Eval.Frequency_Weighted_Intersection_over_Union()
                    PC = Eval.Mean_Precision()
                    logger.info("########## Eval{} ############".format(name))

                    logger.info('\nEpoch:{:.3f}, {} PA1:{:.3f}, MPA1:{:.3f}, MIoU1:{:.3f}, FWIoU1:{:.3f}, PC:{:.3f}'.
                                format(epoch, name, PA, MPA, MIoU, FWIoU, PC))
                    Eval.Print_Every_class_Eval(logger=logger)

                    writer.add_scalar('PA' + name, PA, epoch)
                    writer.add_scalar('MPA' + name, MPA, epoch)
                    writer.add_scalar('MIoU' + name, MIoU, epoch)
                    writer.add_scalar('FWIoU' + name, FWIoU, epoch)
                    return PA, MPA, MIoU, FWIoU

                PA_src, MPA_src, MIoU_src, FWIoU_src = val_info(eval, "source")
                logger.info('\tCurrent source PA: {}, MPA: {}, MIoU: {}, FWIoU: {},'.format(PA_src, MPA_src, MIoU_src, FWIoU_src))

            inters_over_union_classes = per_class_iu(hist)
            all_res[i_iter] = inters_over_union_classes
            pickle_dump(all_res, cache_path)
        else:
            inters_over_union_classes = all_res[i_iter]

        computed_miou = round(np.nanmean(inters_over_union_classes) * 100, 2)
        if cur_best_miou < computed_miou:
            cur_best_miou = computed_miou
            cur_best_model = restore_from

        logger.info('\tCurrent best model:', cur_best_model)
        logger.info('\tCurrent best mIoU:', cur_best_miou)
        if verbose:
            display_stats(cfg, test_loader.dataset.class_names, inters_over_union_classes, logger)


def load_checkpoint_for_evaluation(model, checkpoint, device):
    saved_state_dict = torch.load(checkpoint)
    model.load_state_dict(saved_state_dict)
    model.eval()
    model.cuda(device)


def display_stats(cfg, name_classes, inters_over_union_classes, logger=None):
    for ind_class in range(cfg.NUM_CLASSES):
        if logger:
            logger.info(name_classes[ind_class]
                        + '\t' + str(round(inters_over_union_classes[ind_class] * 100, 2)))
        print(name_classes[ind_class]
              + '\t' + str(round(inters_over_union_classes[ind_class] * 100, 2)))


def decode_labels(mask, num_images=1, num_classes=19):
    """Decode batch of segmentation masks.

    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict.

    Returns:
      A batch with num_images RGB images of the same size as the input.
    """
    with open(DEFAULT_INFO_PATH, 'r') as fp:
        infor = json.load(fp)
    label_colours = infor["palette"]
    if isinstance(mask, torch.Tensor):
        mask = mask.data.cpu().numpy()
    n, h, w = mask.shape
    if n < num_images:
        num_images = n
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        img = np.zeros((h, w, 3))
        for j_, j in enumerate(mask[i, :, :]):
            for k_, k in enumerate(j):
                if k < num_classes:
                    img[int(j_), int(k_), :] = label_colours[int(k)]
        outputs[i] = img
    return torch.from_numpy(outputs.transpose([0, 3, 1, 2]).astype('float32')).div_(255.0)


def init_logger(cfg):
    # logger configure
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(os.path.join(cfg.TRAIN.TENSORBOARD_LOGDIR, cfg.name + "_log.txt"))
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
