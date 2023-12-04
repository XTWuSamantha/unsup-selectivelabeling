# %%
import os
os.environ["USL_MODE"] = "USL"
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import random
import numpy as np
import torch
import models.resnet_cifar_cld as resnet_cifar_cld
import models.resnet_cifar as resnet_cifar
import models.resnet_medmnist as resnet_medmnist
import utils
from utils import cfg, logger, print_b
import medmnist
from medmnist import INFO, Evaluator

utils.init(default_config_file="configs/BloodMNIST_ours_reg.yaml")

logger.info(cfg)

print_b("Loading model")

checkpoint = torch.load(cfg.MODEL.PRETRAIN_PATH)

assert cfg.DATASET.NAME in [
    "cifar10", "cifar100", 'bloodmnist', 'dermamnist'], f"{cfg.DATASET.NAME} is not cifar10 or cifar100 or specific MedMNIST"
cifar100 = cfg.DATASET.NAME == "cifar100"
cifar10 = cfg.DATASET.NAME == "cifar10"

if cifar100 or cifar10:
    num_classes = 100 if cifar100 else 10
    # model
    model = resnet_cifar_cld.__dict__[cfg.MODEL.ARCH]().cuda()
    state_dict = utils.single_model(checkpoint["train_model"])
    mismatch = model.load_state_dict(state_dict, strict=False)
    logger.warning(
        f"Key mismatches: {mismatch} (extra contrastive keys are intended)")
    model.eval()    
    # dataset
    print_b("Loading dataset")
    train_memory_dataset, train_memory_loader = utils.train_memory_cifar(
    root_dir=cfg.DATASET.ROOT_DIR,
    batch_size=cfg.DATALOADER.BATCH_SIZE,
    workers=cfg.DATALOADER.WORKERS, transform_name=cfg.DATASET.TRANSFORM_NAME, cifar100=cifar100, stl10 = False)
    target = torch.tensor(train_memory_dataset.targets)
    target.shape

else:
    # model
    model = resnet_medmnist.__dict__[cfg.MODEL.ARCH]().cuda()
    state_dict = utils.single_model(checkpoint["train_model"])
    mismatch = model.load_state_dict(state_dict, strict=False)
    logger.warning(
        f"Key mismatches: {mismatch} (extra contrastive keys are intended)")
    model.eval()    
    # dataset
    print_b("Loading dataset")
    info = INFO[cfg.DATASET.NAME]
    num_classes = len(info['label'])
    train_memory_dataset, train_memory_loader = utils.train_memory_medmnist(
        dataname = cfg.DATASET.NAME,
        batch_size=cfg.DATALOADER.BATCH_SIZE,
        workers=cfg.DATALOADER.WORKERS, transform_name=cfg.DATASET.TRANSFORM_NAME)
    target = []
    for i in train_memory_loader:
        temp = i[1]
        n = temp.numpy()
        for t in range(len(n)):
            target = np.append(target,n[t])
    targetnp = target.astype(int)
    target = torch.tensor(targetnp)
 
# %%
print_b("Loading feat list")
feats_list = utils.get_feats_list(
    model, train_memory_loader, recompute=cfg.RECOMPUTE_ALL, force_no_extra_kwargs=True)

# %%
print_b("Calculating first order kNN density estimation")
d_knns, ind_knns = utils.partitioned_kNN(
    feats_list, K=cfg.USL.KNN_K, recompute=cfg.RECOMPUTE_ALL)
neighbors_dist = d_knns.mean(dim=1)
score_first_order = 1/neighbors_dist

# %%
num_centroids, final_sample_num = utils.get_sample_info_cifar(class_num=cfg.CLASS_NUM,
    chosen_sample_num=cfg.USL.NUM_SELECTED_SAMPLES)
logger.info("num_centroids: {}, final_sample_num: {}".format(
    num_centroids, final_sample_num))

# %%
recompute_num_dependent = cfg.RECOMPUTE_ALL or cfg.RECOMPUTE_NUM_DEP
cluster_labels = np.load(cfg.CLUSTERING_LABEL)

print_b("Getting selections with regularization")
selected_inds, selected_scores = utils.get_selection(utils.get_selection_with_reg_spice, feats_list, neighbors_dist,
                                                        cluster_labels, num_centroids,
                                                        final_sample_num=final_sample_num, iters=cfg.USL.REG.NITERS,
                                                        w=cfg.USL.REG.W,
                                                        momentum=cfg.USL.REG.MOMENTUM,
                                                        horizon_dist=cfg.USL.REG.HORIZON_DIST, alpha=cfg.USL.REG.ALPHA,
                                                        verbose=True, 
                                                        recompute=recompute_num_dependent, 
                                                        a = cfg.USL.A,
                                                        save=True)

counts = np.bincount(target[selected_inds])


print("Class counts:", sum(counts > 0))
print(counts.tolist())

print("max: {}, min: {}".format(counts.max(), counts.min()))

print("Number of selected indices:", len(selected_inds))
print("Selected IDs:")
print(repr(selected_inds))

