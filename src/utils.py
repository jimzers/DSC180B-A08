import os
import random
import math
import pickle
import numpy as np
import tree
import matplotlib.pyplot as plt
import io
import imageio
from PIL import Image
from acme import wrappers
from dm_env import specs

from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam

IMG_HEIGHT = 256
IMG_WIDTH = 256

"""
From utils.py
"""


def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p


def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
        
def get_kinematics(physics, geom_nams, joint_names, actuator_names):
    geom_positions = physics.named.data.geom_xpos[geom_nams]
    joint_angles = physics.named.data.qpos[joint_names]
    joint_velocities = physics.named.data.qvel[joint_names]
    actuator_forces = physics.named.data.actuator_force[actuator_names]

    return {
        'geom_positions': geom_positions,
        'joint_angles': joint_angles,
        'joint_velocities': joint_velocities,
        'actuator_forces': actuator_forces
    }

def get_geoms(physics, geom_nams):
    geom_positions = physics.named.data.geom_xpos[geom_nams]

    return {
        'geom_positions': geom_positions,
    }


def create_stitched_img(env, num_cams=2, img_height=256, img_width=256):
    tmp_img_arr = []

    # stitch all the views together
    for i in range(num_cams):
        img = env.physics.render(img_height, img_width, camera_id=i)
        im = Image.fromarray(img)
        # save temporarily
        tmp_img_arr.append(im)

    # stitch the images together
    # get the width and height of the images
    widths, heights = zip(*(i.size for i in tmp_img_arr))
    # get the total width and height
    total_width = sum(widths)
    max_height = max(heights)
    # create a new image
    new_im = Image.new('RGB', (total_width, max_height))
    # paste the images together
    x_offset = 0
    for im in tmp_img_arr:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    return new_im


def get_flat_obs(time_step):
    # flatten all of the observations into a single array
    flat_obs = tree.flatten(time_step.observation)
    flat_obs[0] = flat_obs[0].reshape(-1,1)[0]
    # combine all of the observations into a single array
    obs = np.concatenate(flat_obs)
    return obs


def save_video(img_arr, video_name='video.mp4', fps=30):
    """
    Save a video from a list of images
    :param img_arr: list of images
    :type img_arr: list
    :param video_name: name of the video
    :type video_name: str
    :param fps: frames per second
    :type fps: int
    :return: True if successful
    :rtype: bool
    """
    video_path = os.path.join(os.getcwd(), video_name)
    writer = imageio.get_writer(video_path, fps=30)
    for img in img_arr:
        writer.append_data(np.array(img))
    writer.close()
    return True


# create hook function to record activations
def hook_fn(module, input, output):
    print('hook_fn called')
    print('layer name:', module.__class__.__name__)
    # print('input:', input)
    print('output:', output.shape)


def named_hook_fn(name):
    def hook_fn(module, input, output):
        print('hook_fn called')
        print('layer class:', module.__class__.__name__)
        print('name:', name)
        # print('input:', input)
        print('output:', output.shape)

    return hook_fn


def recordtodict_hook(name, hook_dict):
    def hook_fn(module, input, output):
        # append to the corresponding list
        hook_dict[name] += output.clone().detach()
        # hook_dict[name] += output.detach()

    return hook_fn


# add hooks
def test_add_hooks(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module.register_forward_hook(named_hook_fn(name))


# initialize a hook_dict that contains an empty list for each layer
def init_hook_dict(model):
    hook_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            hook_dict[name] = []
    return hook_dict


# compile the hook dict into a dict of numpy arrays
def compile_hook_dict(hook_dict):
    compiled_hook_dict = {}
    for name, hook_list in hook_dict.items():
        if len(hook_list) > 0:
            compiled_hook_dict[name] = torch.stack(hook_list, dim=0).detach().cpu().numpy()
    return compiled_hook_dict


# save the hook_dict to a file
def save_hook_dict(hook_dict, save_path):
    # compile the hook_dict
    compiled_hook_dict = compile_hook_dict(hook_dict)
    # save the compiled_hook_dict
    np.save(save_path, compiled_hook_dict)


# load the hook_dict from a file
def load_hook_dict(load_path):
    compiled_hook_dict = np.load(load_path, allow_pickle=True).item()
    return compiled_hook_dict


def clear_hook_dict(hook_dict):
    # clears the items in hook dict in-place
    for name, hook_list in hook_dict.items():
        hook_list.clear()


# save the PCA
def save_pca_dict(pca_dict, save_path):
    np.save(save_path, pca_dict)


# load the PCA
def load_pca_dict(load_path):
    pca_dict = np.load(load_path, allow_pickle=True).item()
    return pca_dict


def get_activations(pca_dict, compiled_hook_dict, layer_name, num_components=2):
    # get the activations
    activations = compiled_hook_dict[layer_name]
    # get the pca
    pca = pca_dict[layer_name]
    # get the transformed activations
    transformed_activations = pca.transform(activations)
    return transformed_activations


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


def plot_single_point(point, activations, pca, layer_name=None):
    transformed = pca.transform(point)
    # make a scatterplot
    fig, ax = plt.subplots(figsize=(10, 10))
    _ = ax.scatter(activations[:, 0], activations[:, 1], s=1, alpha=0.5)
    # overlay our current dot
    _ = ax.scatter(transformed[:, 0], transformed[:, 1], s=100, alpha=1, c='r')

    if layer_name is not None:
        _ = ax.set_title(layer_name)

    # get the image
    fig_im = fig2img(fig)
    fig_im = fig_im.resize((IMG_HEIGHT, IMG_WIDTH))

    plt.close()

    return fig_im


# # convert a matplotlib figure to an image
# def fig2img(fig):
#     # draw the renderer
#     fig.canvas.draw()
#
#     # Get the RGBA buffer from the figure
#     w, h = fig.canvas.get_width_height()
#     buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
#     buf.shape = (w, h, 3)
#
#     # # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
#     # buf = np.roll(buf, 3, axis=2)
#     return buf

def fig2img(fig):
    """
    Convert a Matplotlib figure to a PIL Image and return it
    https://stackoverflow.com/a/61754995
    """
    buf = io.BytesIO()
    _ = fig.savefig(buf)
    _ = buf.seek(0)
    img = Image.open(buf)
    img = img.convert('RGB')
    return img


# environment wrappers
class NormilizeActionSpecWrapper(wrappers.EnvironmentWrapper):
    """Turn each dimension of the actions into the range of [-1, 1]."""

    def __init__(self, environment):
        super().__init__(environment)

        action_spec = environment.action_spec()
        self._scale = action_spec.maximum - action_spec.minimum
        self._offset = action_spec.minimum

        minimum = action_spec.minimum * 0 - 1.
        maximum = action_spec.minimum * 0 + 1.
        self._action_spec = specs.BoundedArray(
            action_spec.shape,
            action_spec.dtype,
            minimum,
            maximum,
            name=action_spec.name)

    def _from_normal_actions(self, actions):
        actions = 0.5 * (actions + 1.0)  # a_t is now in the range [0, 1]
        # scale range to [minimum, maximum]
        return actions * self._scale + self._offset

    def step(self, action):
        action = self._from_normal_actions(action)
        return self._environment.step(action)

    def action_spec(self):
        return self._action_spec


class MujocoActionNormalizer(wrappers.EnvironmentWrapper):
    """Rescale actions to [-1, 1] range for mujoco physics engine.

    For control environments whose actions have bounded range in [-1, 1], this
      adaptor rescale actions to the desired range. This allows actor network to
      output unscaled actions for better gradient dynamics.
    """

    def __init__(self, environment, rescale='clip'):
        super().__init__(environment)
        self._rescale = rescale

    def step(self, action):
        """Rescale actions to [-1, 1] range before stepping wrapped environment."""
        if self._rescale == 'tanh':
            scaled_actions = tree.map_structure(np.tanh, action)
        elif self._rescale == 'clip':
            scaled_actions = tree.map_structure(lambda a: np.clip(a, -1., 1.), action)
        else:
            raise ValueError('Unrecognized scaling option: %s' % self._rescale)
        return self._environment.step(scaled_actions)
