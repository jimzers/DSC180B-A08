"""
Functions for extracting activations from networks.
"""
import torch
import os
import io
import numpy as np
import matplotlib.pyplot as plt
import imageio
from PIL import Image


IMG_HEIGHT = 256
IMG_WIDTH = 256

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


# check if half-ceetah is on the ground by a manual height check
def is_on_ground(env, geom1_name='ground', geom2_name=set(('ffoot', 'bfoot'))):
    # geom1_name typically is ground contact..
    geom1_name = env.physics.model.name2id(geom1_name, 'geom')

    for contact in np.array(env.physics.data.contact):
        geom_names = [env.physics.model.id2name(g, 'geom')
                      for g in [contact.geom1, contact.geom2]]

        # technically if you just have two elements (1 from geom1, and 1 from geom2) then you can just do an intersection and check to see if size is preserved...

        # terrible O(n2)
        for g_name in geom2_name:
            # note: seems to be that ground geom is always 0
            if g_name in geom_names:
                #                 print('contact between {} and {}'.format(*geom_names))
                return True
    return False

