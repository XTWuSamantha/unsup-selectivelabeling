# %%
import os
os.environ["USL_MODE"] = "RANDOM"

from utils import cfg, logger, print_b
import utils
import models.resnet_cifar_cld as resnet_cifar_cld
import models.resnet_medmnist as resnet_medmnist
import medmnist
from medmnist import INFO, Evaluator

import torch
import numpy as np

utils.init(default_config_file="configs/BloodMNIST_random.yaml")

logger.info(cfg)

# %%
print_b("Loading dataset")
assert cfg.DATASET.NAME in [
    "cifar10", "cifar100", 'bloodmnist', 'dermamnist'], f"{cfg.DATASET.NAME} is not cifar10 or cifar100 or specific MedMNIST"
cifar100 = cfg.DATASET.NAME == "cifar100"
cifar10 = cfg.DATASET.NAME == "cifar10"

if cifar100 or cifar10:
    num_classes = 100 if cifar100 else 10
    train_memory_dataset, train_memory_loader = utils.train_memory_cifar(
    root_dir=cfg.DATASET.ROOT_DIR,
    batch_size=cfg.DATALOADER.BATCH_SIZE,
    workers=cfg.DATALOADER.WORKERS, transform_name=cfg.DATASET.TRANSFORM_NAME, cifar100=cifar100, stl10 = False)
    target = torch.tensor(train_memory_dataset.targets)
    target.shape

else:
    info = INFO[cfg.DATASET.NAME]
    num_classes = len(info['label'])
    train_memory_dataset, train_memory_loader = utils.train_memory_medmnist(dataname=cfg.DATASET.NAME,
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
def get_selection_fn(data_len, seed, final_sample_num):
    np.random.seed(seed)
    selected_inds = np.random.choice(np.arange(data_len), size=final_sample_num, replace=False)
    return selected_inds

# %%
num_labeled = cfg.RANDOM.NUM_SELECTED_SAMPLES
data_len = len(train_memory_dataset)
recompute_num_dependent = cfg.RECOMPUTE_ALL or cfg.RECOMPUTE_NUM_DEP

# %%
for seed in cfg.RANDOM.SEEDS:
    print_b(f"Running random selection with seed {seed}")

    selected_inds = utils.get_selection(get_selection_fn, data_len=data_len, final_sample_num=num_labeled, recompute=recompute_num_dependent, save=True, seed=seed, pass_seed=True)

    counts = np.bincount(target[selected_inds])

    print("Class counts:", sum(counts > 0))
    print(counts.tolist())

    print("max: {}, min: {}".format(counts.max(), counts.min()))

    print("Number of selected indices:", len(selected_inds))
    print("Selected IDs:")
    print(repr(selected_inds))

# %%
