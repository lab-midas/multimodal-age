"""This file provides useful methods which do not belong to a specific class"""
import os
import json
import datetime
import gin
import sys
from torchsummary import summary
import torch
import torch.nn as nn
import logging
import wandb
from models.model_factory import ModelFactory
from utils import utils_misc



def gen_run_folder(directory='', run_id ='', use_wandb = False, direct_evaluation = False):
    """
    This method generates a folder for each run in a sweep.
    """
    run_paths = dict()
    run_paths['path_model_id'] = os.path.join(directory, run_id)
    run_paths['path_ckpts_train'] = os.path.join(run_paths['path_model_id'], 'ckpts', 'model.pt')
    run_paths['path_ckpts_train_ft'] = os.path.join(run_paths['path_model_id'], 'ckpts', 'model_ft.pt')
    if use_wandb:
        run_paths['path_wand_config'] = os.path.join(run_paths['path_model_id'], 'wandb_config')
    run_paths['path_summary'] = os.path.join(run_paths['path_model_id'], 'model_summary.txt')
    run_paths['images'] = os.path.join(run_paths['path_model_id'], 'images')
    if direct_evaluation:
        run_paths['results'] = os.path.join(run_paths['path_model_id'], 'results')
    
    # Create folders
    for k, v in run_paths.items():
        if any([x in k for x in ['path_model', 'path_wand', 'images', 'results']]):
            if not os.path.exists(v):
                os.makedirs(v, exist_ok=True)
        elif any([x in k for x in ['path_summary', 'path_ckpts']]):
            if not os.path.exists(v):
                os.makedirs(os.path.dirname(v), exist_ok=True)
                with open(v, 'a'):
                    pass  # atm file creation is sufficient
    

    return run_paths

def gen_eval_folder(path_model_id = '', organ_name = '', use_wandb = False):
    """
    This method generates a folder for each run in a sweep.
    """
    eval_paths = dict()
    if not os.path.isdir(path_model_id):
        path_model_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'experiments'))
        date_creation = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S-%f')
        sweep_id = 'eval_' + date_creation
        if path_model_id and organ_name:
            sweep_id += '_' + path_model_id + '_' + organ_name
        eval_paths['path_model_id'] = os.path.join(path_model_root, sweep_id)
    else:
        eval_paths['path_model_id'] = path_model_id

    eval_paths['results'] = os.path.join(eval_paths['path_model_id'], 'results')
    eval_paths['path_summary'] = os.path.join(eval_paths['path_model_id'], 'model_summary.txt')
    eval_paths['path_logs_eval'] = os.path.join(eval_paths['path_model_id'], 'logs', 'run.log')
    eval_paths['path_gin'] = os.path.join(eval_paths['path_model_id'], 'config_operative.gin')
    if use_wandb:
        eval_paths['path_wand_config'] = os.path.join(eval_paths['path_model_id'], 'wandb_config')

    # Create folders
    for k, v in eval_paths.items():
        if any([x in k for x in ['path_model', 'results', 'path_wand']]):
            if not os.path.exists(v):
                os.makedirs(v, exist_ok=True)
        elif any([x in k for x in ['path_summary', 'path_logs', 'path_gin']]):
            if not os.path.exists(v):
                os.makedirs(os.path.dirname(v), exist_ok=True)
                with open(v, 'a'):
                    pass

    return eval_paths

def gen_sweep_folder(model_id = '', organ_name = ''):
    """
    This method generates a sweep folder structure
    """
    sweep_paths = dict()
    path_model_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'experiments'))
    date_creation = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S-%f')
    sweep_id = 'sweep_' + date_creation
    if model_id and organ_name:
        sweep_id += '_' + model_id + '_' + organ_name
    sweep_paths['path_model_id'] = os.path.join(path_model_root, sweep_id)

    sweep_paths['path_logs_train'] = os.path.join(sweep_paths['path_model_id'], 'logs', 'run.log')
    sweep_paths['path_gin'] = os.path.join(sweep_paths['path_model_id'], 'config_operative.gin')

    # Create folders
    for k, v in sweep_paths.items():
        if any([x in k for x in ['path_model']]):
            if not os.path.exists(v):
                os.makedirs(v, exist_ok=True)
        elif any([x in k for x in ['path_logs', 'path_gin']]):
            if not os.path.exists(v):
                os.makedirs(os.path.dirname(v), exist_ok=True)
                with open(v, 'a'):
                    pass

    return sweep_paths

def save_wandb_config(config, run_paths):
    """
    This method stores the given wandb configuration in the run folder, 
    enabling checkpoint restoration from a hyperparameter tuning sweep.
    """
    dict_config = dict(config)
    file = os.path.join(run_paths['path_wand_config'],'wandb_config.json')
    with open(file, "w") as f:
        json.dump(dict_config, f, indent=4)


def save_config(path_gin, config):
    """
    This method is used to store the config.gin.
    """
    with open(path_gin, 'w') as f_config:
        f_config.write(config)

def print_model_summary_multimodal(model, input_size_image, input_size_tabular, input_size_head, file_path):
    """
    This is a helper method which enables to write e.g. the model summary into a file instead of printing it to the console.
    """
    components = [
        (nn.Sequential(model.module.encoder_imaging, model.module.projector_imaging), input_size_image),
        (nn.Sequential(model.module.encoder_tabular, model.module.projector_tabular), input_size_tabular),
        (model.module.prediction_head, input_size_head)
    ]

    with open(file_path, 'a') as f:
        original_stdout = sys.stdout
        sys.stdout = f
        for component, input_size in components:
            summary(component, input_size)
        sys.stdout = original_stdout

def print_model_summary_unimodal(model, input_size_encoder, input_size_head, file_path):
    """
    This is a helper method which enables to write e.g. the model summary into a file instead of printing it to the console.
    """
    components = [
        (nn.Sequential(model.module.encoder, model.module.projector), input_size_encoder),
        (model.module.prediction_head, input_size_head)
    ]


    with open(file_path, 'a') as f:
        original_stdout = sys.stdout
        sys.stdout = f
        for component, input_size in components:
            summary(component, input_size)
        sys.stdout = original_stdout

@gin.configurable
def get_model_compare_parameter(config_path_model_1, checkpoint_path_model_1, config_path_model_2, checkpoint_path_model_2):
    """
    This method serves as a wrapper for retrieving relevant parameters defined in config.gin.
    Since main cannot be declared as @gin.configurable, this method is necessary to access the required parameters in main.
    """
    return config_path_model_1, checkpoint_path_model_1, config_path_model_2, checkpoint_path_model_2


def init_wandb(factory, config, name_of_run, sweep_paths, use_wandb, direct_evaluation, set_config = True, eval = False):
    """
    This method initializes wandb.
    """
    wandb.login(key = config['wandb_key'])
    run = wandb.init(project=config['project_name'], config = config, name = name_of_run)
    config = wandb.config
    if set_config:
        factory.set_config(config=config)
    if eval == True:
        run_paths = gen_eval_folder(path_model_id=factory.get_encoder_name()) 
    else:
        run_paths = gen_run_folder(directory=sweep_paths['path_model_id'], run_id=wandb.run.name, use_wandb=use_wandb, direct_evaluation = direct_evaluation)
    save_wandb_config(config=config, run_paths=run_paths)
    return run_paths, config

@gin.configurable
def gen_start(encoder_name, organ_name, direct_evaluation, hyperparameter_tuning, use_wandb, device, config_path = None):
    """
    This method initializes the model factory and creates the folders.
    """
    factory  = ModelFactory(encoder_name=encoder_name, device = device)
    sweep_paths = gen_sweep_folder(model_id=factory.get_encoder_name(), organ_name = organ_name)
    utils_misc.set_loggers(sweep_paths['path_logs_train'], logging.INFO)
    save_config(sweep_paths['path_gin'], gin.config_str())
    config = factory.load_config(hyperparameter_tuning=hyperparameter_tuning, use_wandb=use_wandb, config_path=config_path)
    return factory, sweep_paths, config


@gin.configurable
def gen_start_eval(encoder_name, organ_name, device, config_path = None):
    """
    This method initializes the model factory and creates the folders for the evaluation.
    """
    factory  = ModelFactory(encoder_name=encoder_name, device = device)
    config = factory.load_config(hyperparameter_tuning=False, use_wandb=False, config_path=config_path)
    eval_paths = gen_eval_folder(path_model_id=factory.get_encoder_name(), organ_name=organ_name) 
    utils_misc.set_loggers(eval_paths['path_logs_eval'], logging.INFO)
    save_config(eval_paths['path_gin'], gin.config_str())
    return factory, eval_paths, config

@gin.configurable
def load_model(model, checkpoint_path = None, device = None, pred_head = False, old_model = False):
    """
    This method loads a model.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if pred_head:
        allowed_prefixes = [
            'module.encoder_imaging', 
            'module.encoder_tabular', 
            'module.prediction_head',
            'module.encoder',
            'module.projector',
            'module.projector_imaging', 
            'module.projector_tabular',
        ]
    else:
        allowed_prefixes = [
            'module.encoder_imaging', 
            'module.projector_imaging', 
            'module.encoder_tabular', 
            'module.projector_tabular',
            'module.encoder',
            'module.projector'
        ]
    remapped_checkpoint = {}
    for k, v in checkpoint.items():
        if any(k.startswith(prefix) for prefix in allowed_prefixes):
            new_key = k
            remapped_checkpoint[new_key] = v

    load_result = model.load_state_dict(remapped_checkpoint, strict=False)
    logging.info(f"Missing keys: {load_result.missing_keys}")
    logging.info(f"Unexpected keys: {load_result.unexpected_keys}")

def print_model(model, config, run_paths, data_type, tabular_data_size, in_channels, organ):
    """
    This method prints the model structure.
    """
    size = get_organ_size(organ, in_channels)
    if data_type == "multimodal":
        if config.use_ph:
            if config.two_label == True:
                extra_bit = 0
                if config.extra_bit == True:
                    extra_bit = 1
                print_model_summary_multimodal(model, size, (tabular_data_size,), (config.projection_dim + extra_bit,), file_path=run_paths['path_summary'])
            else:
                print_model_summary_multimodal(model, size, (tabular_data_size,), (config.projection_dim + config.projection_dim,), file_path=run_paths['path_summary'])
        else:
            if config.two_label == True:
                extra_bit = 0
                if config.extra_bit == True:
                    extra_bit = 1
                print_model_summary_multimodal(model, size, (tabular_data_size,), (config.hidden_dim_image_ph + extra_bit,), file_path=run_paths['path_summary'])
            else:
                print_model_summary_multimodal(model, size, (tabular_data_size,), (config.hidden_dim_image_ph + config.hidden_dim_tabular_ph,), file_path=run_paths['path_summary'])
    elif data_type == "image":
        if config.use_ph:
            print_model_summary_unimodal(model, size, (config.projection_dim,), file_path=run_paths['path_summary'])
        else:
            print_model_summary_unimodal(model, size, (config.hidden_dim_image_ph,), file_path=run_paths['path_summary'])
    else:
        if config.use_ph:
            print_model_summary_unimodal(model, (tabular_data_size,), (config.projection_dim,), file_path=run_paths['path_summary'])
        else:
            print_model_summary_unimodal(model, (tabular_data_size,), (config.hidden_dim_tabular_ph,), file_path=run_paths['path_summary'])


def get_organ_dict():
    """
    This method returns a dictionary with organ spcific values.
    """
    return {
            "brain": [get_brain_paths, 1],
            "heart": [get_heart_paths, 1],
            "kidney": [get_kidney_paths, 4],
            "liver": [get_liver_paths, 4],
            "spleen": [get_spleen_paths, 4],
            "pancreas": [get_pancreas_paths, 4]
        }

@gin.configurable
def get_brain_paths(image_file, key_csv_path_training, key_csv_path_test, training):
    """
    Wrapper function for brain specific paths.
    """
    if training:
        return image_file, key_csv_path_training
    else:
        return image_file, key_csv_path_test

@gin.configurable
def get_heart_paths(image_file, key_csv_path_training, key_csv_path_test, training):
    """
    Wrapper function for heart specific paths.
    """
    if training:
        return image_file, key_csv_path_training
    else:
        return image_file, key_csv_path_test

@gin.configurable
def get_spleen_paths(image_file, key_csv_path_training, key_csv_path_test, training):
    """
    Wrapper function for heart specific paths.
    """
    if training:
        return image_file, key_csv_path_training
    else:
        return image_file, key_csv_path_test

@gin.configurable
def get_liver_paths(image_file, key_csv_path_training, key_csv_path_test, training):
    """
    Wrapper function for heart specific paths.
    """
    if training:
        return image_file, key_csv_path_training
    else:
        return image_file, key_csv_path_test

@gin.configurable
def get_kidney_paths(image_file, key_csv_path_training, key_csv_path_test, training):
    """
    Wrapper function for heart specific paths.
    """
    if training:
        return image_file, key_csv_path_training
    else:
        return image_file, key_csv_path_test

@gin.configurable
def get_pancreas_paths(image_file, key_csv_path_training, key_csv_path_test, training):
    """
    Wrapper function for heart specific paths.
    """
    if training:
        return image_file, key_csv_path_training
    else:
        return image_file, key_csv_path_test

def get_organ_columns_csv_path(organ_name, data_type):
    """
    Wrapper function for used columns csv.
    """
    if data_type == "tabular":
        return f"./configs/columns_{organ_name}_image_biomarkers.csv"      # CHANGE HERE
    else:
        #return f"./configs/columns_{organ_name}_tabular_general.csv"	# change back!!
        return f"./configs/columns_{organ_name}_{data_type}.csv"



def get_image_suffix(image_file):
    """
    This method returns a dictionary with organ suffixes for dataset builder.
    """
    image_organ_suffix_mapping = {
        "brain": "",
        "heart": "_sa",
        "liver": "",
        "spleen": "",
        "kidney": "",
        "pancreas": "",
    }

    for key, value in image_organ_suffix_mapping.items():
        if key in image_file:   # check if the key exists in the string
            return value

    return ""

def get_organ_size(organ, in_channels):
    """
    This method returns a dictionary with organ spcific values for printing the model structure.
    """
    image_size = {
            "brain": (in_channels, 182, 218, 182),
            "heart": (in_channels, 72, 76, 50),
            "liver": (in_channels, 120, 100, 70),
            "spleen":(in_channels, 60, 60, 50),
            "pancreas": (in_channels, 80, 50, 50),
            "kidney": (in_channels, 40, 40, 50)
        }
    if organ not in image_size:
        raise Exception(f"Organ {organ} not in organ-imagesize mapping")
    return image_size.get(organ)


def forward_wrapper(model, modality):
    """
    Wrapper function for IntegratedGradients calculation.
    """
    def wrapper(*args):
        if modality == "multimodal":
            images, tabular = args
            batch = {"image": images, "tabular_data": tabular}
        elif modality == "image":
            (images,) = args
            batch = {"image": images}
        elif modality == "tabular":
            (tabular,) = args
            batch = {"tabular_data": tabular}
        else:
            raise Exception(f"Unknown modality {modality}")

        return model(batch)

    return wrapper


