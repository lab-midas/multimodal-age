"""File for dataset builder"""

import pandas as pd
from torch.utils.data import Dataset
from input_pipeline.utils import utils
import h5py
import logging
import torch
import numpy as np
import gin
from utils.utils_params import get_image_suffix

@gin.configurable
class BioBankDataset(Dataset):
    """
    This class is used as dataset builder to provide the data.
    """
    def __init__(self, image_file, key_csv_path, tabular_data_path, label_csv_path, organ_columns_csv_path, organ_name, data_type="multimodal", transforms=None, two_label=False, train_only_same_age_range=False):
        
        self.data_type = data_type

        if (train_only_same_age_range == True and two_label == False and data_type == "multimodal"):
            raise Exception("Training with same age range is only possible with two labels.")

        self.image_file = image_file
        self.two_label = two_label
        self.transforms = transforms
        self.only_same_age_range = train_only_same_age_range
        self.organ_name = organ_name

        image_organs = ["brain", "heart"]
        self.keys = pd.read_csv(key_csv_path, header=None, names=['eid'])
        self.utils = utils(tabular_data_path = tabular_data_path, organ_columns_csv_path = organ_columns_csv_path, label_csv_path = label_csv_path, keys = self.keys, data_type = self.data_type, two_label=two_label, only_same_age_range=train_only_same_age_range)

        if self.data_type == "multimodal" or self.data_type == "tabular":
            self.tabular_data, self.keys = self.utils.load_tabular_data()
            self.tabular_feature_names = self.tabular_data.columns.tolist()
            self.first_assessment_labels = self.utils.get_first_assessment_age_labels()

        self.labels, self.keys = self.utils.load_age_labels()

        if self.only_same_age_range:
            if len(self.keys) >= len(self.tabular_data):
                self.length = len(self.tabular_data)
            else:
                self.length = len(self.keys)
        else:
            self.length = len(self.keys)

        if self.organ_name == "kidney" and self.data_type != "tabular":
            self.length = self.length * 2

        self.keys = self.keys.sort_values(by='eid').reset_index(drop=True) 
        if self.data_type != "tabular":
            self.image_suffix = get_image_suffix(image_file)
            if self.organ_name in image_organs:
                self.contrasts = ["image"]
            else:
                self.contrasts = ['fat', 'inp', 'opp', 'wat']


    def __len__(self):
        """
        Returns the total number of images.
        """
        return self.length
    
    def __getitem__(self, idx):
        """
        Returns the image, tabular data and label at the given index.
        """
        if self.organ_name == "kidney":
            side = "l" if idx % 2 == 0 else "r"
            idx = idx//2

        key = self.keys.iloc[idx]['eid']
        if self.data_type == "multimodal" or self.data_type == "image":
            if self.organ_name == "kidney":
                image_file = list(self.image_file)
                image_file[-26] = side 
                image_file = ''.join(image_file)
            else:
                image_file = self.image_file
            with h5py.File(image_file, 'r', swmr = True) as f:
                images = []
                for contrast in self.contrasts:
                    images.append(self.load_image(f, contrast, key, self.image_suffix))

                image = np.stack(images, axis=0)
                if image.shape[0] == 1:
                    image = np.squeeze(image, axis=0)
                    if image.ndim == 4:
                        image = np.squeeze(image[:, :, np.random.randint(np.shape(image)[2], size=1), :])
                    image = np.expand_dims(image, axis=0)
                
                if self.transforms:
                    image = np.expand_dims(image, axis=0)
                    image = self.transforms(image=image)['image'][0]
                
                image = torch.from_numpy(image)
                    
        if self.organ_name == "kidney":
            key = f"{key}_{side}"
        
        if self.data_type == "multimodal" or self.data_type == "tabular":
            tabular_data = torch.tensor(self.tabular_data.iloc[idx].to_numpy(), dtype=torch.float32)
        label = torch.tensor(self.labels.iloc[idx]['age'], dtype=torch.float32)
        if self.data_type == "image":
            sample = {'image': image.float(), 'label': label, 'key': key}

        elif self.data_type == "multimodal":
            if self.two_label == True:
                label_t = torch.tensor(self.first_assessment_labels.iloc[idx]['21003-0.0'], dtype=torch.float32)
                sample = {'image': image.float(), 'tabular_data': tabular_data, 'label': label, 'label-t': label_t, 'key': key}
            else:
                sample = {'image': image.float(), 'tabular_data': tabular_data, 'label': label, 'key': key}

        else:
            #label = torch.tensor(self.first_assessment_labels.iloc[idx]['21003-0.0'], dtype=torch.float32)
            label = torch.tensor(self.labels.iloc[idx]['age'], dtype=torch.float32)
            sample = {'tabular_data': tabular_data, 'label': label, 'key': key}
        return sample

    def get_names(self):
        """
        This function returns the names of the tabular columns.
        """
        return self.tabular_feature_names

    def get_tabular_size(self):
        """
        This method returns amount of columns in the tabular data.
        """
        return self.utils.get_tabular_size()

    def load_image(self, f, contrast, key, suffix):
        """
        This method loads a specific image for the dataset.
        """
        if contrast == "image":
            candidates = [f"{key}_2{suffix}", f"{key}_3{suffix}"]
        else:
            candidates = [f"{key}{suffix}"]

        for candidate in candidates:
            if candidate in f[contrast]:
                return f[contrast][candidate][()]

        raise KeyError(f"No image found for contrast='{contrast}', key='{key}', suffixes={candidates}")




    
