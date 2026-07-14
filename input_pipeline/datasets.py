"""This modul creates, loads, preprocesses and augments datasets."""

import gin
from input_pipeline.dataset_builder import BioBankDataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch
import logging
from batchgenerators.transforms.color_transforms import GammaTransform
from batchgenerators.transforms.spatial_transforms import MirrorTransform
from batchgenerators.transforms.crop_and_pad_transforms import RandomCropTransform
from batchgenerators.transforms.abstract_transforms import Compose
from utils.utils_params import get_organ_dict, get_organ_columns_csv_path

@gin.configurable
def load_train_data(name, config, data_type, organ_name, batch_size = 64):
    """
    This function creates or loads the dataset.
    """

    if name == "biobank":

        organ_path_dict = get_organ_dict()

        function, _ = organ_path_dict.get(organ_name)
        image_file, key_csv_path = function(training = True)
        if data_type != "multimodal":
            modality = "unimodal"
        else:
            modality = data_type
        organ_columns_csv_path = get_organ_columns_csv_path(organ_name, data_type)

        g = torch.Generator()
        g.manual_seed(1)

        if data_type == "multimodal":
            two_label = config.two_label
            train_only_same_age_range = config.train_only_same_age_range
        else:
            two_label = False
            train_only_same_age_range = False

        transforms = []
        if getattr(config, "data_augmentation", None):
            logging.info("Data augmentation is enabled for training dataset.")
            transforms.append(RandomCropTransform(crop_size=config.crop_size[organ_name], data_key='image', margins=config.crop_margins[organ_name]))
            transforms.append(GammaTransform(gamma_range=config.gamma_range, data_key='image'))
            transforms.append(MirrorTransform(axes=[config.mirror_axis[organ_name]], data_key='image'))
            train_transforms = Compose(transforms)
        else:
            train_transforms = None

        full_dataset = BioBankDataset(
            image_file = image_file, 
            key_csv_path = key_csv_path, 
            organ_columns_csv_path = organ_columns_csv_path, 
            organ_name = organ_name, data_type = data_type, 
            transforms = train_transforms, 
            two_label = two_label, 
            train_only_same_age_range = train_only_same_age_range
            )

        train_size = int(0.7 * len(full_dataset))  
        val_size = len(full_dataset) - train_size

        logging.info(f"train_size: {train_size}") 
        logging.info(f"val_size: {val_size}")  

        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=g)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g, drop_last = True, pin_memory = True, num_workers = 4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, generator=g, drop_last = True, pin_memory = True, num_workers = 4)

        if data_type != "image":
            tabular_data_size = full_dataset.get_tabular_size()
        else:
            names = None
            tabular_data_size = None

        return train_loader, val_loader, tabular_data_size

    else:
        raise ValueError

@gin.configurable
def load_test_data(name, config, data_type, organ_name, batch_size = 64):
    """
    This function creates or loads the test dataset.
    """

    if name == "biobank":

        organ_path_dict = get_organ_dict()

        function, _ = organ_path_dict.get(organ_name)
        image_file, key_csv_path = function(training = False)
        if data_type != "multimodal":
            modality = "unimodal"
        else:
            modality = data_type
        organ_columns_csv_path = get_organ_columns_csv_path(organ_name, data_type)

        g = torch.Generator()
        g.manual_seed(1)

        if data_type == "multimodal":
            two_label = config.two_label
            train_only_same_age_range = config.train_only_same_age_range
        else:
            two_label = False
            train_only_same_age_range = False

        full_dataset = BioBankDataset(
            image_file = image_file, 
            key_csv_path = key_csv_path, 
            organ_name = organ_name, 
            organ_columns_csv_path = organ_columns_csv_path, 
            data_type = data_type, 
            transforms = None,
            two_label = two_label, 
            train_only_same_age_range = train_only_same_age_range)

        logging.info(f"test_size: {len(full_dataset)}") 

        test_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, generator=g, drop_last = True, pin_memory = True, num_workers = 4)

        if data_type != "image":
            names = full_dataset.get_names()
            tabular_data_size = full_dataset.get_tabular_size()
        else:
            names = None
            tabular_data_size = None


        return test_loader, tabular_data_size, names

    else:
        raise ValueError




    
    


