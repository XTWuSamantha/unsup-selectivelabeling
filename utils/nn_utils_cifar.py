# Datasets
import random
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
from tqdm import tqdm

from .augment import Augment, Cutout
from .config_utils import cfg, logger
from .nn_utils import get_transform, normalization_kwargs_dict
import medmnist
from medmnist import INFO, Evaluator

transform_init = transforms.Compose([
     transforms.ToTensor(),
     transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.201))
 ])

class CIFAR10_LT(datasets.CIFAR10):

    def __init__(self, root, indexs=None, train=True,
                 transform=transform_init, 
                 download=False):
        super(CIFAR10_LT, self).__init__(root, train=train,
                 transform=transform, 
                 download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        self.data = [Image.fromarray(img) for img in self.data]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

def get_sample_info_cifar(class_num, chosen_sample_num):
    num_centroids = class_num
    final_sample_num = chosen_sample_num

    # We get one more centroid to take empty clusters into account
    if chosen_sample_num == 2500:
        num_centroids = 2501
        final_sample_num = 2500
        logger.warning("Returning 2501 as the number of centroids")

    return num_centroids, final_sample_num

def get_sample_info_cifar_usl(chosen_sample_num):
    num_centroids = chosen_sample_num
    final_sample_num = chosen_sample_num

    # We get one more centroid to take empty clusters into account
    if chosen_sample_num == 2500:
        num_centroids = 2501
        final_sample_num = 2500
        logger.warning("Returning 2501 as the number of centroids")

    return num_centroids, final_sample_num

# def get_selection_with_reg(data, neighbors_dist, cluster_labels, num_centroids, final_sample_num, iters=1, w=1, momentum=0.5, horizon_dist=None, alpha=1, verbose=False):
#     selection_regularizer = torch.zeros_like(neighbors_dist)
#     selected_ids_comparison_mask = F.one_hot(
#         cluster_labels, num_classes=num_centroids)
#     for _ in tqdm(range(iters)):
#         selected_inds = []
#         selected_inds_max = []
#         selected_scores = []
#         for cls_ind in range(num_centroids):
#             if len(selected_inds) == final_sample_num:
#                 break
#             match_arr = cluster_labels == cls_ind
#             match = torch.where(match_arr)[0]
#             if len(match) == 0:
#                 continue

#             # Scores in the selection process
#             scores = 1 / neighbors_dist[match_arr] - \
#                 w * selection_regularizer[match_arr]
#             scores_list = scores.tolist()
#             n = int(final_sample_num/num_centroids)
#             min_dist_ind = pd.Series(scores_list).sort_values(ascending = False).index[:n]
#             min_dist_ind_max = scores.argmax()
#             selected_inds_max.append(match[min_dist_ind_max])
#             for i in min_dist_ind:
#                 selected_inds.append(match[i])
#                 selected_scores.append(scores[i])

#         selected_inds = np.array(selected_inds)
#         selected_data = data[selected_inds]
#         selected_inds_max = np.array(selected_inds_max)
#         selected_data_max = data[selected_inds_max]
#         selected_scores = np.array(selected_scores)
#         zipped = zip(selected_inds, selected_scores)
#         sort_zipped = sorted(zipped, key=lambda x: (x[1], x[0]), reverse = True)
#         result = zip(*sort_zipped)
#         selected_inds, selected_scores = [list(x) for x in result]
#         # print(selected_inds_max)
#         # print(selected_inds)
#         # This is square distances: (N_full, N_selected)
#         # data: (N_full, 1, dim)
#         # selected_data: (1, N_selected, dim)
#         new_selection_regularizer = (
#             (data[:, None, :] - selected_data_max[None, :, :]) ** 2).sum(dim=-1)

#         if verbose:
#             logger.info(
#                 f"Max: {new_selection_regularizer.max()} Mean: {new_selection_regularizer.mean()}")

#         # Distance to the instance within the same cluster should be ignored
#         new_selection_regularizer = (1 - selected_ids_comparison_mask) * \
#             new_selection_regularizer + selected_ids_comparison_mask * 1e10

#         assert not torch.any(new_selection_regularizer == 0), "{}".format(
#             torch.where(new_selection_regularizer == 0))

#         if verbose:
#             logger.info(f"Min: {new_selection_regularizer.min()}")

#         # If it is outside of horizon dist (square distance), than we ignore it.
#         if horizon_dist is not None:
#             new_selection_regularizer[new_selection_regularizer >=
#                                       horizon_dist] = 1e10

#         # selection_regularizer: N_full
#         new_selection_regularizer = (
#             1 / new_selection_regularizer ** alpha).sum(dim=1)

#         selection_regularizer = selection_regularizer * \
#             momentum + new_selection_regularizer * (1 - momentum)

#     assert len(
#         selected_inds) == final_sample_num, f"{len(selected_inds)} != {final_sample_num}"
#     return selected_inds, selected_scores


# def get_selection_with_reg(data, neighbors_dist, cluster_labels, num_centroids, final_sample_num, iters=1, w=1, momentum=0.5, horizon_dist=None, alpha=1, verbose=False):
#     selection_regularizer = torch.zeros_like(neighbors_dist)
#     selected_ids_comparison_mask = F.one_hot(
#         cluster_labels, num_classes=final_sample_num)
#     for _ in tqdm(range(iters)):
#         selected_inds = []
#         for cls_ind in range(num_centroids):
#             if len(selected_inds) == final_sample_num:
#                 break
#             match_arr = cluster_labels == cls_ind
#             match = torch.where(match_arr)[0]
#             if len(match) == 0:
#                 continue

#             # Scores in the selection process
#             scores = 1 / neighbors_dist[match_arr] - \
#                 w * selection_regularizer[match_arr]
#             min_dist_ind = scores.argmax()
#             selected_inds.append(match[min_dist_ind])

#         selected_inds = np.array(selected_inds)
#         selected_data = data[selected_inds]
#         # This is square distances: (N_full, N_selected)
#         # data: (N_full, 1, dim)
#         # selected_data: (1, N_selected, dim)
#         new_selection_regularizer = (
#             (data[:, None, :] - selected_data[None, :, :]) ** 2).sum(dim=-1)

#         if verbose:
#             logger.info(
#                 f"Max: {new_selection_regularizer.max()} Mean: {new_selection_regularizer.mean()}")

#         # Distance to the instance within the same cluster should be ignored
#         new_selection_regularizer = (1 - selected_ids_comparison_mask) * \
#             new_selection_regularizer + selected_ids_comparison_mask * 1e10

#         assert not torch.any(new_selection_regularizer == 0), "{}".format(
#             torch.where(new_selection_regularizer == 0))

#         if verbose:
#             logger.info(f"Min: {new_selection_regularizer.min()}")

#         # If it is outside of horizon dist (square distance), than we ignore it.
#         if horizon_dist is not None:
#             new_selection_regularizer[new_selection_regularizer >=
#                                       horizon_dist] = 1e10

#         # selection_regularizer: N_full
#         new_selection_regularizer = (
#             1 / new_selection_regularizer ** alpha).sum(dim=1)

#         selection_regularizer = selection_regularizer * \
#             momentum + new_selection_regularizer * (1 - momentum)

#     assert len(
#         selected_inds) == final_sample_num, f"{len(selected_inds)} != {final_sample_num}"
#     return selected_inds

def get_selection_with_reg(data, neighbors_dist, cluster_labels, num_centroids, final_sample_num, iters=1, w=1, momentum=0.5, horizon_dist=None, alpha=1, verbose=False):
    selection_regularizer = torch.zeros_like(neighbors_dist)
    selected_ids_comparison_mask = F.one_hot(
        cluster_labels, num_classes=num_centroids)
    # selected_ids_comparision_mask (50000*10)
    for _ in tqdm(range(iters)):
        selected_inds = []
        selected_inds_max = []
        selected_scores = []
        for cls_ind in range(num_centroids):
            if len(selected_inds) == final_sample_num:
                break
            match_arr = cluster_labels == cls_ind
            match = torch.where(match_arr)[0]
            if len(match) == 0:
                continue

            # Scores in the selection process
            scores = 1 / neighbors_dist[match_arr] - \
                w * selection_regularizer[match_arr]
            scores_list = scores.tolist()
            n = int(final_sample_num/num_centroids)
            min_dist_ind = pd.Series(scores_list).sort_values(ascending = False).index[:n]
            min_dist_ind_max = scores.argmax()
            selected_inds_max.append(match[min_dist_ind_max])
            for i in min_dist_ind:
                selected_inds.append(match[i])
                selected_scores.append(scores[i])

        selected_inds = np.array(selected_inds)
        selected_data = data[selected_inds]
        selected_inds_max = np.array(selected_inds_max)
        selected_data_max = data[selected_inds_max]
        selected_scores = np.array(selected_scores)
        zipped = zip(selected_inds, selected_scores)
        sort_zipped = sorted(zipped, key=lambda x: (x[1], x[0]), reverse = True)
        result = zip(*sort_zipped)
        selected_inds, selected_scores = [list(x) for x in result]

        new_selection_regularizer = (
            (data[:, None, :] - selected_data_max[None, :, :]) ** 2).sum(dim=-1)
        

        if verbose:
            logger.info(
                f"Max: {new_selection_regularizer.max()} Mean: {new_selection_regularizer.mean()}")

        # Distance to the instance within the same cluster should be ignored
        # inter
        new_selection_regularizer = (1 - selected_ids_comparison_mask) * \
            new_selection_regularizer + selected_ids_comparison_mask * 1e10
        
        # all samples
        # new_selection_regularizer = new_selection_regularizer + selected_ids_comparison_mask * 1e10

        # all samples with different weight(need to be tested)
        # new_selection_regularizer = new_selection_regularizer + selected_ids_comparison_mask * (1e11* new_selection_regularizer + 1e10)

        # intra
        # new_selection_regularizer = selected_ids_comparison_mask * new_selection_regularizer +  1e10

        assert not torch.any(new_selection_regularizer == 0), "{}".format(
            torch.where(new_selection_regularizer == 0))

        if verbose:
            logger.info(f"Max: {new_selection_regularizer.max()} Mean: {new_selection_regularizer.mean()} Min: {new_selection_regularizer.min()}")

        # If it is outside of horizon dist (square distance), than we ignore it.
        if horizon_dist is not None:
            new_selection_regularizer[new_selection_regularizer >=
                                      horizon_dist] = 1e10

        # selection_regularizer: N_full
        new_selection_regularizer = (
            1 / new_selection_regularizer ** alpha).sum(dim=1)

        selection_regularizer = selection_regularizer * \
            momentum + new_selection_regularizer * (1 - momentum)

    assert len(
        selected_inds) == final_sample_num, f"{len(selected_inds)} != {final_sample_num}"
    return selected_inds, selected_scores


# low density
def get_selection_with_reg_low(data, neighbors_dist, cluster_labels, num_centroids, final_sample_num, iters=1, w=1, momentum=0.5, horizon_dist=None, alpha=1, verbose=False):
    selection_regularizer = torch.zeros_like(neighbors_dist)
    selected_ids_comparison_mask = F.one_hot(
        cluster_labels, num_classes=num_centroids)
    for _ in tqdm(range(iters)):
        selected_inds = []
        selected_inds_max = []
        selected_scores = []
        for cls_ind in range(num_centroids):
            if len(selected_inds) == final_sample_num:
                break
            match_arr = cluster_labels == cls_ind
            match = torch.where(match_arr)[0]
            if len(match) == 0:
                continue

            # Scores in the selection process
            # 这里对scores进行了更改，同时把max改成了min
            scores = 1 / neighbors_dist[match_arr] + \
                w * selection_regularizer[match_arr]
            scores_list = scores.tolist()
            n = int(final_sample_num/num_centroids)
            min_dist_ind = pd.Series(scores_list).sort_values(ascending = True).index[:n]
            min_dist_ind_max = scores.argmin()
            selected_inds_max.append(match[min_dist_ind_max])
            for i in min_dist_ind:
                selected_inds.append(match[i])
                selected_scores.append(scores[i])

        selected_inds = np.array(selected_inds)
        selected_data = data[selected_inds]
        selected_inds_max = np.array(selected_inds_max)
        selected_data_max = data[selected_inds_max]
        selected_scores = np.array(selected_scores)
        zipped = zip(selected_inds, selected_scores)
        sort_zipped = sorted(zipped, key=lambda x: (x[1], x[0]), reverse = True)
        result = zip(*sort_zipped)
        selected_inds, selected_scores = [list(x) for x in result]
        # print(selected_inds_max)
        # print(selected_inds)
        # This is square distances: (N_full, N_selected)
        # data: (N_full, 1, dim)
        # selected_data: (1, N_selected, dim)
        new_selection_regularizer = (
            (data[:, None, :] - selected_data_max[None, :, :]) ** 2).sum(dim=-1)
        

        if verbose:
            logger.info(
                f"Max: {new_selection_regularizer.max()} Mean: {new_selection_regularizer.mean()}")

        # Distance to the instance within the same cluster should be ignored
        # inter
        new_selection_regularizer = (1 - selected_ids_comparison_mask) * \
            new_selection_regularizer + selected_ids_comparison_mask * 1e10
        
        # all samples
        # new_selection_regularizer = new_selection_regularizer + selected_ids_comparison_mask * 1e10

        # all samples with different weight(need to be tested)
        # new_selection_regularizer = new_selection_regularizer + selected_ids_comparison_mask * (1e11* new_selection_regularizer + 1e10)

        # intra
        # new_selection_regularizer = selected_ids_comparison_mask * new_selection_regularizer +  1e10

        assert not torch.any(new_selection_regularizer == 0), "{}".format(
            torch.where(new_selection_regularizer == 0))

        if verbose:
            logger.info(f"Min: {new_selection_regularizer.min()}")

        # If it is outside of horizon dist (square distance), than we ignore it.
        if horizon_dist is not None:
            new_selection_regularizer[new_selection_regularizer >=
                                      horizon_dist] = 1e10

        # selection_regularizer: N_full
        new_selection_regularizer = (
            1 / new_selection_regularizer ** alpha).sum(dim=1)

        selection_regularizer = selection_regularizer * \
            momentum + new_selection_regularizer * (1 - momentum)

    assert len(
        selected_inds) == final_sample_num, f"{len(selected_inds)} != {final_sample_num}"
    return selected_inds, selected_scores


def get_selection_with_reg_iter(data, neighbors_dist, cluster_labels, num_centroids, final_sample_num, iters=1, w=1, momentum=0.5, horizon_dist=None, alpha=1, verbose=False):
    # print(cluster_labels)
    selection_regularizer = torch.zeros_like(neighbors_dist)
    selected_ids_comparison_mask = F.one_hot(
        cluster_labels, num_classes=num_centroids)
    # print(selected_ids_comparison_mask)
    selected_inds_final = []
    n = int(final_sample_num/num_centroids)
    for _ in tqdm(range(iters)):
        selected_inds = []
        for cls_ind in range(num_centroids):
            if len(selected_inds) == final_sample_num:
                break
            match_arr = cluster_labels == cls_ind
            match = torch.where(match_arr)[0]
            if len(match) == 0:
                continue

            # Scores in the selection process
            scores = 1 / neighbors_dist[match_arr] - \
                w * selection_regularizer[match_arr]
            min_dist_ind = scores.argmax()
            selected_inds.append(match[min_dist_ind])

        selected_inds = np.array(selected_inds)
        selected_data = data[selected_inds]
        # This is square distances: (N_full, N_selected)
        # data: (N_full, 1, dim)
        # selected_data: (1, N_selected, dim)
        new_selection_regularizer = (
            (data[:, None, :] - selected_data[None, :, :]) ** 2).sum(dim=-1)

        if verbose:
            logger.info(
                f"Max: {new_selection_regularizer.max()} Mean: {new_selection_regularizer.mean()}")

        # Distance to the instance within the same cluster should be ignored
        new_selection_regularizer = (1 - selected_ids_comparison_mask) * \
            new_selection_regularizer + selected_ids_comparison_mask * 1e10

        assert not torch.any(new_selection_regularizer == 0), "{}".format(
            torch.where(new_selection_regularizer == 0))

        if verbose:
            logger.info(f"Min: {new_selection_regularizer.min()}")

        # If it is outside of horizon dist (square distance), than we ignore it.
        if horizon_dist is not None:
            new_selection_regularizer[new_selection_regularizer >=
                                      horizon_dist] = 1e10

        # selection_regularizer: N_full
        new_selection_regularizer = (
            1 / new_selection_regularizer ** alpha).sum(dim=1)

        selection_regularizer = selection_regularizer * \
            momentum + new_selection_regularizer * (1 - momentum)
    
        selected_inds_final.append(selected_inds)
        print(selected_inds_final)

    # assert len(
    #     selected_inds) == final_sample_num, f"{len(selected_inds)} != {final_sample_num}"
    return selected_inds

def get_selection_with_reg_spice(data, neighbors_dist, cluster_labels, num_centroids, final_sample_num, iters=1, w=1, momentum=0.5, horizon_dist=None, alpha=1, a=0, verbose=False):
    n = int(final_sample_num/num_centroids)
    # print(cluster_labels)
    four_hot_data = np.zeros((len(cluster_labels), final_sample_num))
    for i, category in enumerate(cluster_labels):
        start_idx = category * n
    end_idx = (category + 1) * n
    four_hot_data[i, start_idx:end_idx] = 1
    cluster_labels = torch.tensor(cluster_labels)
    four_hot_data = torch.tensor(four_hot_data)
    selection_regularizer = torch.zeros_like(neighbors_dist)
    selected_ids_comparison_mask = F.one_hot(
        cluster_labels, num_classes=num_centroids)

    for _ in tqdm(range(iters)):
        selected_inds = []
        selected_inds_max = []
        selected_scores = []
        # 在每个cluster中选出scores最高的n个样本
        for cls_ind in range(num_centroids):
            if len(selected_inds) == final_sample_num:
                break
            match_arr = cluster_labels == cls_ind
            match = torch.where(match_arr)[0]
            if len(match) == 0:
                continue

            # Scores in the selection process
            scores = 1 / neighbors_dist[match_arr] - \
                w * selection_regularizer[match_arr]
            # scores = 1 / neighbors_dist[match_arr]
            scores_list = scores.tolist()
            min_dist_ind = pd.Series(scores_list).sort_values(ascending = False).index[:n]
            min_dist_ind_max = scores.argmax()
            selected_inds_max.append(match[min_dist_ind_max])
            for i in min_dist_ind:
                selected_inds.append(match[i])
                selected_scores.append(scores[i])
        

        selected_inds = np.array(selected_inds)
        selected_data = data[selected_inds]
        # print(selected_data.size())
        # # tensor[40, 128]
        selected_inds_max = np.array(selected_inds_max)
        selected_data_max = data[selected_inds_max]
        selected_scores = np.array(selected_scores)
        # print(selected_scores)
        zipped = zip(selected_inds, selected_scores)
        sort_zipped = sorted(zipped, key=lambda x: (x[1], x[0]), reverse = True)
        result = zip(*sort_zipped)
        selected_inds, selected_scores = [list(x) for x in result]

        new_selection_regularizer = (
            (data[:, None, :] - selected_data[None, :, :]) ** 2).sum(dim=-1)

        if verbose:
            logger.info(
                f"Max: {new_selection_regularizer.max()} Mean: {new_selection_regularizer.mean()}")

        # Distance to the instance within the same cluster should be ignored
        # inter
        # new_selection_regularizer = (1 - selected_ids_comparison_mask) * \
        #     new_selection_regularizer + selected_ids_comparison_mask * 1e10

        # new_selection_regularizer = (1 - four_hot_data) * \
        #     new_selection_regularizer + four_hot_data * 1e10
        
        # all samples
        # new_selection_regularizer = new_selection_regularizer + selected_ids_comparison_mask * 1e10
        # alpha = 0
        # new_selection_regularizer = new_selection_regularizer
        # new_selection_regularizer[new_selection_regularizer == 0] = 1e10


        # all samples with different weight(need to be tested)
               
        new_selection_regularizer =(1 + a * four_hot_data) * new_selection_regularizer
        new_selection_regularizer[new_selection_regularizer == 0] = 1e10


        # intra
        # new_selection_regularizer = selected_ids_comparison_mask * new_selection_regularizer +  1e10

        # new_selection_regularizer = four_hot_data * new_selection_regularizer + (1 - four_hot_data)* 1e10
        # new_selection_regularizer[new_selection_regularizer == 0] = 1e10


        assert not torch.any(new_selection_regularizer == 0), "{}".format(
            torch.where(new_selection_regularizer == 0))

        if verbose:
            logger.info(f"Max: {new_selection_regularizer.max()} Mean: {new_selection_regularizer.mean()} Min: {new_selection_regularizer.min()}")

        # If it is outside of horizon dist (square distance), than we ignore it.
        if horizon_dist is not None:
            new_selection_regularizer[new_selection_regularizer >=
                                      horizon_dist] = 1e10


        # selection_regularizer: N_full
        new_selection_regularizer = (
            1 / new_selection_regularizer ** alpha).sum(dim=1)
        
        selection_regularizer = selection_regularizer * \
            momentum + new_selection_regularizer * (1 - momentum)

    assert len(
        selected_inds) == final_sample_num, f"{len(selected_inds)} != {final_sample_num}"
    return selected_inds, selected_scores


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class PRETRAIN_CIFAR10(datasets.CIFAR10):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dict: {'image': image, 'target': index of target class, 'meta': dict}
        """
        img, target = self.data[index], self.targets[index]
        img_size = (img.shape[0], img.shape[1])
        img = Image.fromarray(img)
        class_name = self.classes[target]

        if self.transform is not None:
            img = self.transform(img)

        out = {'image': img, 'target': target, 'meta': {
            'im_size': img_size, 'index': index, 'class_name': class_name}}

        return out


class PRETRAIN_CIFAR100(datasets.CIFAR100):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dict: {'image': image, 'target': index of target class, 'meta': dict}
        """
        img, target = self.data[index], self.targets[index]
        img_size = (img.shape[0], img.shape[1])
        img = Image.fromarray(img)
        class_name = self.classes[target]

        if self.transform is not None:
            img = self.transform(img)

        out = {'image': img, 'target': target, 'meta': {
            'im_size': img_size, 'index': index, 'class_name': class_name}}

        return out


def train_dataset_cifar(transform_name):
    if transform_name == "FixMatch-cifar10" or transform_name == "SCAN-cifar10" or transform_name == "FixMatch-cifar100" or transform_name == "SCAN-cifar100":
        normalization_kwargs = normalization_kwargs_dict[transform_name]
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            Augment(4),
            transforms.ToTensor(),
            transforms.Normalize(**normalization_kwargs),
            Cutout(
                n_holes=1,
                length=16,
                random=True)])
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(**normalization_kwargs)])

        train_transforms = {
            'standard': transform_val, 'augment': transform_train}
    elif transform_name == "CLD-cifar10" or transform_name == "CLD-cifar100":
        # CLD uses MoCov2's aug: similar to SimCLR
        normalization_kwargs = normalization_kwargs_dict[transform_name]
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**normalization_kwargs),
        ])

        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(**normalization_kwargs)])
    else:
        raise ValueError(f"Unsupported transform type: {transform_name}")

    if cfg.DATASET.NAME == "cifar10":
        # Transform is set on the wrapper if load_local_global_dataset is True
        train_dataset_cifar = datasets.CIFAR10(
            root=cfg.DATASET.ROOT_DIR, train=True, transform=transform_train, download=True)

        val_dataset = datasets.CIFAR10(
            root=cfg.DATASET.ROOT_DIR, train=False, transform=transform_val, download=True)
    elif cfg.DATASET.NAME == "cifar100":
        train_dataset_cifar = datasets.CIFAR100(
            root=cfg.DATASET.ROOT_DIR, train=True, transform=transform_train, download=True)

        val_dataset = datasets.CIFAR100(
            root=cfg.DATASET.ROOT_DIR, train=False, transform=transform_val, download=True)

    return train_dataset_cifar, val_dataset


# Memory bank on CIFAR
def train_memory_cifar(root_dir, cifar100, stl10, transform_name, batch_size=128, workers=2, with_val=False):
    # Note that CLD uses the same normalization for CIFAR 10 and CIFAR 100

    transform_test = get_transform(transform_name)

    if cifar100:
        train_memory_dataset = datasets.CIFAR100(root=root_dir, train=True,
                                                 download=True, transform=transform_test)
        if with_val:
            val_memory_dataset = datasets.CIFAR100(root=root_dir, train=False,
                                                   download=True, transform=transform_test)
    if stl10:
        train_memory_dataset = datasets.STL10(root=root_dir, split = "train",
                                                download=True, transform=transform_test)
        if with_val:
            val_memory_dataset = datasets.STL10(root=root_dir, split = "test",
                                                   download=True, transform=transform_test)

    else:
        train_memory_dataset = datasets.CIFAR10(root=root_dir, train=True,
                                                download=True, transform=transform_test)
        if with_val:
            val_memory_dataset = datasets.CIFAR10(root=root_dir, train=False,
                                                  download=True, transform=transform_test)

    train_memory_loader = torch.utils.data.DataLoader(
        train_memory_dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, drop_last=False)

    if with_val:
        val_memory_loader = torch.utils.data.DataLoader(
            val_memory_dataset, batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True, drop_last=False)
        return train_memory_dataset, train_memory_loader, val_memory_dataset, val_memory_loader
    else:
        return train_memory_dataset, train_memory_loader


def train_memory_medmnist(dataname, transform_name, batch_size=128, workers=2, with_val=False):
    # Note that CLD uses the same normalization for CIFAR 10 and CIFAR 100

    transform_test = get_transform(transform_name)
    info = INFO[dataname]
    DataClass = getattr(medmnist, info['python_class'])
    train_memory_dataset =  DataClass(split='train', transform=transform_test, download=True)
    if with_val:
        val_memory_dataset = DataClass(split='train+val', transform=transform_test, download=True)

    train_memory_loader = torch.utils.data.DataLoader(
        train_memory_dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, drop_last=False)

    if with_val:
        val_memory_loader = torch.utils.data.DataLoader(
            val_memory_dataset, batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True, drop_last=False)
        return train_memory_dataset, train_memory_loader, val_memory_dataset, val_memory_loader
    else:
        return train_memory_dataset, train_memory_loader
