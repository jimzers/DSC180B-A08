"""
Functions for analysis of kinematic and activation data.
"""
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import rsatoolbox
import sys


def plot_activations(activations, layer_name=None, save_path=None, show=False):
    # assumes 2 components
    # grab x and y
    x, y = activations[:, 0], activations[:, 1]
    # plot the activations
    fig, ax = plt.subplots(figsize=(10, 10))
    _ = ax.scatter(x, y, s=1)
    _ = ax.set_xlabel('PC1')
    _ = ax.set_ylabel('PC2')
    if layer_name is not None:
        ax.set_title(layer_name)
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()

    return fig


def plot_activations_umap(activations, layer_name=None, save_path=None, show=False):
    # assumes 2 components
    reducer = umap.UMAP()
    scaled_activations = StandardScaler().fit_transform(activations)
    reducer.fit(scaled_activations)
    embedding = reducer.transform(scaled_activations)

    # grab x and y
    x, y = embedding[:, 0], embedding[:, 1]
    # plot the activations
    fig, ax = plt.subplots(figsize=(10, 10))
    _ = ax.scatter(x, y, s=1)
    _ = ax.set_xlabel('UM1')
    _ = ax.set_ylabel('UM2')

    # color the points on viridis scale by the timestep position out of 1000
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, 1000))
    for i in range(1000):
        _ = ax.scatter(x[i::1000], y[i::1000], s=1, color=colors[i])

    if layer_name is not None:
        ax.set_title(layer_name)
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()

    return fig, reducer  # you can reuse the reducer to transform future activations


def cka(X, Y):
    """
    Implementation of CKA similarity index as formulated by Kornblith et al.,(2019)
    """

    # making a copy prevents modifying original arrays
    X = X.copy()
    Y = Y.copy()

    # center x and y first to do dot product
    X -= X.mean(axis=0)
    Y -= Y.mean(axis=0)

    # dot products
    x_xt = X.T.dot(X)
    y_yt = Y.T.dot(Y)
    x_yt = Y.T.dot(X)

    # Frobenius norm = root of the sum of squares of the entries when X and Y are centered
    return (x_yt ** 2).sum() / np.sqrt((x_xt ** 2).sum() * (y_yt ** 2).sum())
    # return np.linalg.norm(x=x_yt, ord='fro') / (np.linalg.norm(x=x_xt, ord='fro') * np.linalg.norm(x=y_yt, ord='fro'))


def plot_cka_kinematics(loaded_hook_dict, total_kinematic_dict, save_path=None):
    """
    plot cka similarity matrix between activations and kinematic features
    """
    # part b
    figure_5b = {'activation': [],
                 'kinematic_feature': [],
                 'cka': []}

    for i in total_kinematic_dict:
        total_kinematic_dict[i] = np.array(total_kinematic_dict[i])

    # nested loop through each combination and add to the dictionary
    # if its geom_positions, reshape it from 3d to 2d

    # get combinations between kinematics and activations
    for feat in total_kinematic_dict.keys():
        for activation in loaded_hook_dict.keys():
            if feat == 'geom_positions':
                x_3d = total_kinematic_dict[feat]
                nsamples, nx, ny = x_3d.shape
                x = x_3d.reshape((nsamples, nx * ny))
                cka_calc = cka(loaded_hook_dict[activation], x)

                figure_5b['activation'].append(activation)
                figure_5b['kinematic_feature'].append(feat)
                figure_5b['cka'].append(cka_calc)
            else:
                cka_calc = cka(loaded_hook_dict[activation], total_kinematic_dict[feat])

                figure_5b['activation'].append(activation)
                figure_5b['kinematic_feature'].append(feat)
                figure_5b['cka'].append(cka_calc)

    df_b = pd.DataFrame(figure_5b).drop_duplicates().pivot('kinematic_feature', 'activation', 'cka')

    # force the range of the heatmap to be between 0 and 1
    plot_b = sns.heatmap(df_b, cbar_kws={'label': 'Feature encoding (CKA)'}, cmap="Blues", vmin=0, vmax=1)
    plt.savefig(save_path)

    return plot_b


def plot_cka_activations(loaded_hook_dict, save_path='src/viz/cka_activations.png'):
    """
    plot cka similarity matrix between activation layers
    """
    # part c
    figure_5c = {'activation_1': [],
                 'activation_2': [],
                 'cka': []}

    # get combinations between activations
    for activation1 in loaded_hook_dict.keys():
        for activation2 in loaded_hook_dict.keys():
            cka_calc = cka(loaded_hook_dict[activation1], loaded_hook_dict[activation2])
  
            figure_5c['cka'].append(cka_calc)
            figure_5c['activation_1'].append(activation1)
            figure_5c['activation_2'].append(activation2)

    df_c = pd.DataFrame(figure_5c).pivot('activation_1', 'activation_2', 'cka')
    plot_c = sns.heatmap(df_c, cbar_kws={'label': 'Representational similarity (CKA)'}, cmap="Blues")
    plt.savefig(save_path)
    return plot_c

def plot_cka_activations_between_models(loaded_hook_dict, loaded_hook_dict2, save_path='src/viz/cka_activations_compare.png'):
    """
    plot cka similarity matrix between activation layers
    """
    # part c
    figure_5c = {'activation_1': [],
                 'activation_2': [],
                 'cka': []}

    # get combinations between activations
    for activation1 in loaded_hook_dict.keys():
        for activation2 in loaded_hook_dict2.keys():
            cka_calc = cka(loaded_hook_dict[activation1], loaded_hook_dict2[activation2])
      
            figure_5c['cka'].append(cka_calc)
            figure_5c['activation_1'].append(activation1)
            figure_5c['activation_2'].append(activation2)

    df_c = pd.DataFrame(figure_5c).pivot('activation_1', 'activation_2', 'cka')
   
    plot_c = sns.heatmap(df_c, cbar_kws={'label': 'Representational similarity (CKA)'}, cmap='Blues')
    plt.savefig(save_path)
    return plot_c


def plot_rsa(activation, kinematic, total_kinematic_dict, loaded_hook_dict,
             save_path='src/viz/rsa.png'):
    """
    plot rsa similarity between kinematic feature and activation layer
    activation: 'mean_linear' or 'log_std_linear'
    kinematic: 'joint_angles', 'joint_velocities', 'actuator_forces'
    """
    kmeans = KMeans(n_clusters=50, random_state=0).fit(total_kinematic_dict[kinematic])
    df = pd.DataFrame(total_kinematic_dict[kinematic])
    df['cluster_label'] = pd.Series(kmeans.labels_)

    all_calcs = []
    b = loaded_hook_dict[activation]

    for i in range(50):
        a = df[df['cluster_label'] == i].drop(['cluster_label'], axis=1).values
        c = list(np.dot(a, b.T))
        all_calcs += c

    out = np.array(all_calcs)
    data = rsatoolbox.data.Dataset(out)
    rdms = rsatoolbox.rdm.calc_rdm(data)
    title = activation + ' vs. ' + kinematic
    rsatoolbox.vis.show_rdm(rdms, rdm_descriptor=title, show_colorbar='panel', figsize=(8, 8))
    plt.savefig(save_path)
