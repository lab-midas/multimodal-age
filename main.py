import torch

from input_pipeline import datasets
import gin
from models.model_factory import ModelFactory
from evaluation.eval import test_chkpt, test_chkpt_compare
import wandb

from train import Trainer
from absl import app, flags

from evaluation.eval import evaluate
from utils import utils_params
import logging

from captum.attr import IntegratedGradients
from types import SimpleNamespace

import pandas as pd

FLAGS = flags.FLAGS

flags.DEFINE_boolean('train', True, 'Specify whether to train or evaluate a model.')
flags.DEFINE_boolean('only_finetuning', False, 'Specify if you only want to finetune a model.')
flags.DEFINE_boolean('gradients', False, 'Specify if you only want to get the models importances.')
flags.DEFINE_boolean('compare_embeddings', False, 'Specify if you only want to compare models.')
flags.DEFINE_boolean('hyperparameter_tuning', False, 'Specify whether hyperparametertuning should be enabled')
flags.DEFINE_boolean('use_wandb', True, 'Specify whether weights and biases should be used or not')
flags.DEFINE_boolean('direct_evaluation', True, 'Specify whether the model should be evaluated directly after the training. This is not possible in combination with hyperparameter optimization')
flags.DEFINE_string('data_type', 'tabular', 'Specifies the used data type')
flags.DEFINE_string('organ', 'brain', 'Specifies the used organ')

gin.parse_config_file('./configs/config.gin')


def main(argv):
    possible_organs = utils_params.get_organ_dict()
    if FLAGS.organ not in possible_organs:
        raise Exception(f"Organ {FLAGS.organ} not supported")
    _, in_channels = possible_organs.get(FLAGS.organ)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if FLAGS.data_type != "multimodal" and FLAGS.data_type != "image" and FLAGS.data_type != "tabular":
        raise Exception("datatype not supported")
    
    if FLAGS.train:        
        factory, sweep_paths, config = utils_params.gen_start(FLAGS.data_type, FLAGS.organ, FLAGS.direct_evaluation, FLAGS.hyperparameter_tuning, FLAGS.use_wandb, device, None)
        logging.info(f"Device: {device}")

        def train_model(config = None, name_of_run = None):
            """
            Training function. Initalizes wandb if neccessary and starts the training afterwards. Furthermore, specific run folder are generated.
            """
            if FLAGS.use_wandb:
                run_paths, config = utils_params.init_wandb(factory, config, name_of_run, sweep_paths, FLAGS.use_wandb, FLAGS.direct_evaluation)
            else:
                run_paths = utils_params.gen_run_folder(directory=sweep_paths['path_model_id'], run_id=factory.get_encoder_name(), use_wandb=FLAGS.use_wandb, direct_evaluation = FLAGS.direct_evaluation)

            ds_train, ds_val, tabular_data_size = datasets.load_train_data(config = config, data_type = FLAGS.data_type, organ_name = FLAGS.organ, batch_size=config.batch_size)
            
            factory.set_tabular_input_size(tabular_data_size)
            factory.set_image_input_channels(in_channels)
            model = torch.nn.DataParallel(factory.create_model()).to(device)       

            # Train the model
            utils_params.print_model(model, config, run_paths, FLAGS.data_type, tabular_data_size, in_channels, FLAGS.organ)
            torch.save(model.state_dict(), run_paths['path_ckpts_train'])

            trainer = Trainer(model, ds_train, ds_val, run_paths, config, FLAGS.data_type, FLAGS.use_wandb, device)

            logging.info(f'Starting Pre-Training with {factory.get_encoder_name()} for {config.total_epochs} epochs')
            trainer.train()

            factory.set_mode("prediction")
            trainer.finetune()

            if FLAGS.direct_evaluation:
                ds_test, _, _ = datasets.load_test_data(config = config, data_type = FLAGS.data_type, organ_name = FLAGS.organ, batch_size=config.batch_size)
                evaluate(model, ds_train, device, config, FLAGS.organ, save_paths=run_paths)
                evaluate(model, ds_val, device, config, FLAGS.organ, save_paths=run_paths)
                evaluate(model, ds_test, device, config, FLAGS.organ, save_paths=run_paths)

        if FLAGS.hyperparameter_tuning and FLAGS.use_wandb:
            sweep_id = wandb.sweep(config, project=config.project_name)
            wandb.agent(sweep_id, function=train_model, count=20) 
        elif (not FLAGS.hyperparameter_tuning and FLAGS.use_wandb) or (not FLAGS.hyperparameter_tuning and not FLAGS.use_wandb):
            train_model(config=config,name_of_run=factory.get_encoder_name() + '_' + FLAGS.organ)
        else: 
            raise Exception("Hyperparameter optimization is not possible without wandb")
        wandb.finish()

    elif FLAGS.gradients or FLAGS.only_finetuning:
        if FLAGS.gradients and FLAGS.only_finetuning:
            raise Exception("only finetuning or Integrated Gradient calculation is possible")
        factory, sweep_paths, config = utils_params.gen_start(FLAGS.data_type, FLAGS.organ, FLAGS.direct_evaluation, FLAGS.hyperparameter_tuning, FLAGS.use_wandb, device)
        logging.info(f"Device: {device}")
        if FLAGS.use_wandb:
            if FLAGS.gradients:
                name_of_run = f"{factory.get_encoder_name()}_{FLAGS.organ}_gradients"
            else:
                name_of_run = f"{factory.get_encoder_name()}_{FLAGS.organ}_FT"
            run_paths, config = utils_params.init_wandb(factory, config, name_of_run, sweep_paths, FLAGS.use_wandb, FLAGS.direct_evaluation)
        else:
            run_paths = utils_params.gen_run_folder(directory=sweep_paths['path_model_id'], run_id=factory.get_encoder_name(), use_wandb=FLAGS.use_wandb, direct_evaluation = FLAGS.direct_evaluation)              

        if FLAGS.only_finetuning:
            ds_train, ds_val, tabular_data_size = datasets.load_train_data(config = config, data_type = FLAGS.data_type, organ_name = FLAGS.organ, batch_size=config.batch_size)
        else:
            ds_test, tabular_data_size, tabular_columns_names = datasets.load_test_data(config = config, data_type = FLAGS.data_type, organ_name = FLAGS.organ, batch_size=config.batch_size)

        factory.set_tabular_input_size(tabular_data_size)
        factory.set_image_input_channels(in_channels)

        model = torch.nn.DataParallel(factory.create_model()).to(device)
        utils_params.load_model(model, device = device, pred_head = FLAGS.gradients)

        model.module.freeze_encoder()
        utils_params.print_model(model, config, run_paths, FLAGS.data_type, tabular_data_size, in_channels, FLAGS.organ)
        # test_chkpt(model, ds_val, device, run_paths, FLAGS.data_type, use_wandb = FLAGS.use_wandb)
        model.module.set_mode("prediction")
        
        if FLAGS.only_finetuning:
            
            trainer = Trainer(model, ds_train, ds_val, run_paths, config, FLAGS.data_type, FLAGS.use_wandb, device)
            trainer.finetune()

            if FLAGS.direct_evaluation:
                ds_test, _, _ = datasets.load_test_data(config = config, data_type = FLAGS.data_type, organ_name = FLAGS.organ, batch_size=config.batch_size)
                evaluate(model, ds_test, device, config, FLAGS.organ, save_paths = run_paths)


        else:
            model.eval()

            ig = IntegratedGradients(utils_params.forward_wrapper(model, "multimodal"))

            batch = next(iter(ds_test))
            sample_images = batch['image'].to(device)
            sample_tabular = batch['tabular_data'].to(device)

            sample_images.requires_grad_()
            sample_tabular.requires_grad_()

            attributions = ig.attribute((sample_images, sample_tabular), n_steps=100, internal_batch_size=config.batch_size)
            attribution_image, attribution_tabular = attributions
            tab_attr1 = attribution_tabular.detach().cpu().numpy().mean(axis=0)

            batch = next(iter(ds_test))

            sample_images.requires_grad_()
            sample_tabular.requires_grad_()

            attributions = ig.attribute((sample_images, sample_tabular), n_steps=100, internal_batch_size=config.batch_size)
            attribution_image, attribution_tabular = attributions
            tab_attr2 = attribution_tabular.detach().cpu().numpy().mean(axis=0)
            tab_attr = (tab_attr1 + tab_attr2)/2
            logging.info(f"Importances: {tab_attr}")
            for name, importance in zip(tabular_columns_names, tab_attr):
                logging.info(f"{name}: {importance:.4f}")


    elif FLAGS.compare_embeddings:
        config_path_model_1, checkpoint_path_model_1, config_path_model_2, checkpoint_path_model_2 = utils_params.get_model_compare_parameter()
        factory, sweep_paths, config1 = utils_params.gen_start(FLAGS.data_type, FLAGS.organ, FLAGS.direct_evaluation, FLAGS.hyperparameter_tuning, FLAGS.use_wandb, device, config_path = config_path_model_1)
        config1 = SimpleNamespace(**config1)
        factory.set_config(config=config1)
        logging.info(f"Device: {device}")

        ds_test, tabular_data_size, _ = datasets.load_test_data(config = config1, data_type = FLAGS.data_type, organ_name = FLAGS.organ, batch_size=config1.batch_size)
        factory.set_tabular_input_size(tabular_data_size)
        factory.set_image_input_channels(in_channels)

        model = torch.nn.DataParallel(factory.create_model()).to(device) 
        utils_params.load_model(model, checkpoint_path_model_1, device = device)

        config2 = factory.load_config(hyperparameter_tuning=FLAGS.hyperparameter_tuning, use_wandb=False, config_path=config_path_model_2)

        model2 = torch.nn.DataParallel(factory.create_model()).to(device) 
        utils_params.load_model(model2, checkpoint_path_model_2, device = device)

        if FLAGS.use_wandb:
            def namespace_to_dict(ns):
                return vars(ns) if isinstance(ns, SimpleNamespace) else ns
            project_name = config1.project_name
            del config1.project_name
            del config2.project_name
            combined_config = {
                "project_name": project_name,
                "model 1": namespace_to_dict(config1),
                "model 2": namespace_to_dict(config2)
            }
            run_paths, _ = utils_params.init_wandb(factory, combined_config, name_of_run = f"{factory.get_encoder_name()}_compare", sweep_paths = sweep_paths, use_wandb=FLAGS.use_wandb, direct_evaluation = FLAGS.direct_evaluation, set_config = False)
        else:
            run_paths = utils_params.gen_run_folder(directory=sweep_paths['path_model_id'], run_id=factory.get_encoder_name(), use_wandb=FLAGS.use_wandb, direct_evaluation = FLAGS.direct_evaluation)
    
        test_chkpt_compare(model, model2, ds_test, device, run_paths, FLAGS.data_type, use_wandb = FLAGS.use_wandb)


    else: 
        factory, eval_paths, config = utils_params.gen_start_eval(FLAGS.data_type, FLAGS.organ, device)
        logging.info(f"Device: {device}")
                     
        ds_test, tabular_data_size, _ = datasets.load_test_data(config = config, data_type = FLAGS.data_type, organ_name = FLAGS.organ, batch_size=config.batch_size)
        factory.set_tabular_input_size(tabular_data_size)
        factory.set_image_input_channels(in_channels)    

        model = torch.nn.DataParallel(factory.create_model()).to(device)
        utils_params.load_model(model, device = device, pred_head = True)

        utils_params.print_model(model, config, eval_paths, FLAGS.data_type, tabular_data_size, in_channels, FLAGS.organ)

        model.module.freeze_encoder()
        model.module.set_mode("prediction")

        if FLAGS.data_type == "multimodal":
            two_label = config.two_label
        else:
            two_label = False
        
        evaluate(model, ds_test, device, FLAGS.organ, two_label, save_paths = eval_paths)
            

if __name__ == "__main__":
    app.run(main)


