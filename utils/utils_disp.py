"""This file provides useful methods for displaying embeddings which do not belong to a specific class"""
import logging
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib as mpl
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch.amp import autocast
import numpy as np
import torch
import gin
import requests
import wandb


def display(projection_embeddings, labels, types, epoch, run_paths, method, feature_space = False, use_wandb = False):
    """
    This method calculates a lower dimensional plot of the embeddings and saves it.
    """
    if method == "PCA":
        reducer = PCA(n_components=2)
    else:
        reducer = TSNE(n_components=2, perplexity=20, learning_rate=200, init='pca', random_state=42)

    embeddings_2d = reducer.fit_transform(projection_embeddings)

    if set(types) == {'image'}:
        data_types = [['image']]
    elif set(types) == {'tabular'}:
        data_types = [['tabular']]
    else:
        data_types = [['image'], ['tabular'], ['image', 'tabular']]

    markers = {'image': 'o', 'tabular': 's'}
    colormap = {"image": "YlOrRd","tabular": "GnBu"}


    for data_type_group in data_types:

        fig, ax = plt.subplots(figsize=(10, 8))

        for data_type in data_type_group:
            indices = types == data_type
            sc = ax.scatter(embeddings_2d[indices, 0],
                        embeddings_2d[indices, 1],
                        c=labels[indices],
                        cmap=colormap[data_type],
                        marker=markers[data_type],
                        alpha=0.7,
                        label=f"{data_type} embedding")
            cbar = fig.colorbar(sc, ax=ax, fraction=0.08, pad=0.05)
            cbar.set_label(f'{data_type} labels', fontsize=14)
            cbar.ax.tick_params(labelsize=12)

        ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                       labelbottom=False, labelleft=False)
        plt.tight_layout()
        suffix = '_'.join(data_type_group) + ("_FS" if feature_space else "_LS")
        #plt.savefig(f"{epoch}_{method}_{suffix}.png", dpi=300)
        plt.savefig(f"{run_paths['images']}/{epoch}_{method}_{suffix}.png", dpi=300)
        if use_wandb:
            wandb.log({f"{method}/embedding_{method}_epoch_{epoch}_{suffix}": wandb.Image(plt)}, commit=False)
        plt.close()

def display_compare(embeddings, labels, types, sources, run_paths, use_wandb = False):
    """
    This method can plot the embeddings of two models in one plot for comparision.
    """
    reducer = PCA(n_components=2)
    embeddings_2d = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))
    markers = {'imagemodel1': 'o', 'tabularmodel1': 's', 'imagemodel2': 'v', 'tabularmodel2': '*'}
    colormap = {"image": "YlOrRd","tabular": "GnBu"} 

    for model in np.unique(sources):
        for data_type in ['image', 'tabular']:
            indices = (types == data_type) & (model  == sources)
            # with np.printoptions(threshold=np.inf, linewidth=np.inf):
            #     logging.info(f"indices: {indices}")
            ax.scatter(embeddings_2d[indices, 0],
                        embeddings_2d[indices, 1],
                        c=labels[indices],
                        cmap=colormap[data_type],
                        marker=markers[data_type+model],
                        alpha=0.7,
                        label=f"{data_type} {model}")


    for i, data_type in enumerate(['image', 'tabular']):
        indices = (types == data_type)
        vmin = labels[indices].min()
        vmax = labels[indices].max()

        cbar = plt.colorbar(ScalarMappable(norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax), cmap=colormap[data_type]), ax=ax, fraction=0.08, pad=0.05)
        cbar.set_label(f"{data_type} Labels")

    plt.title('2D Visualization of Image and Tabular Embeddings')
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{run_paths['images']}/model_comaprison.png")
    if use_wandb:
        wandb.log({f"model_comaprison": wandb.Image(plt)}, commit=False)
    plt.close()

def plot_start(model, ds_val, device, run_paths, data_type, two_label, use_wandb = False):
    """
    This method plots the embeddings before training.
    """
    model.eval()
    embeddings_list = []
    labels_list = []
    types_list = []
    j = 0
    with torch.no_grad():  
        for batch in ds_val:  
            if j > 40:
                break  
            labels = batch['label'].to(device)
            with autocast(device_type=device.type):
                z, _ = model(batch)
            if data_type == "multimodal" and two_label == True:
                labels_t = batch['label-t'].to(device)
                gen_lists(z, labels, embeddings_list, labels_list, types_list, data_type, labels_t)
            else:
                gen_lists(z, labels, embeddings_list, labels_list, types_list, data_type)
            j+=1
        
        plot_results(embeddings_list, labels_list, types_list, 0, run_paths, use_wandb)

def log_projection(z_t, z_i, labels):
    """
    This method logs projections.
    """
    for index in range(3):
        image_embedding = z_i[index]
        tabular_embedding = z_t[index]
        label = labels[index]
        logging.info(f"image: {image_embedding}")
        logging.info(f"tabular: {tabular_embedding}")
        logging.info(f"label: {label}")

def plot_results(embeddings_list, labels_list, types_list, epoch, run_paths, sources_list = None, compare = False, feature_space = False, use_wandb = False):
    """
    This method concatinates all given lists for the display function and calls them.
    """
    embeddings_list = np.concatenate(embeddings_list, axis=0)
    labels_list = np.concatenate(labels_list, axis=0)
    types = np.array(types_list)
    if compare:
        if sources_list == None:
            raise Exception("No source list")
        sources = np.array(sources_list)
        display_compare(embeddings_list, labels_list, types, sources, run_paths, use_wandb = use_wandb)
    else:
        display(embeddings_list, labels_list, types, epoch, run_paths, "PCA", feature_space, use_wandb)
        display(embeddings_list, labels_list, types, epoch, run_paths, "TSNE", feature_space, use_wandb)

def gen_lists(z, labels, embeddings_list, labels_list, types_list, data_type = None, labels_t = None):
    """
    This method calcuates all needed lists for the display (returns with call by reference).
    """
    labels_np = labels.detach().cpu().numpy() 
    labels_list.append(labels_np)

    if z.ndim == 3:
        z_i, z_t = z[:, 0, :], z[:, 1, :]
        z_i_np = z_i.detach().cpu().numpy()
        z_t_np = z_t.detach().cpu().numpy()

        embeddings_list.append(z_i_np)
        embeddings_list.append(z_t_np)
        if labels_t is None:
            labels_list.append(labels_np)
        else:
            labels_t_np = labels_t.detach().cpu().numpy() 
            labels_list.append(labels_t_np)

        types_list.extend(['image'] * len(z_i_np))
        types_list.extend(['tabular'] * len(z_t_np))

    else:
        z_np = z.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy() 

        embeddings_list.append(z_np)

        if data_type == "image":
            types_list.extend(['image'] * len(z_np))
        elif data_type == "tabular":
            types_list.extend(['tabular'] * len(z_np))
        else:
            raise Exception("No valid datatype provided")
