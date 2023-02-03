import scipy

from src.activations import *
from src.environment import *
from src.train_cheetah import SAC

# make psd directory
if not os.path.exists('psd'):
    os.makedirs('psd')

# make activations directory
if not os.path.exists('activations'):
    os.makedirs('activations')

# make spike_plots directory
if not os.path.exists('spike_plots'):
    os.makedirs('spike_plots')


# load the environment
env = suite.load(domain_name="cheetah", task_name="run")
# add wrappers onto the environment
env = NormilizeActionSpecWrapper(env)
env = MujocoActionNormalizer(environment=env, rescale='clip')
env = wrappers.SinglePrecisionWrapper(env)


class Args:
    env_name = 'whatever'
    policy = 'Gaussian'
    eval = True
    gamma = 0.99
    tau = 0.005
    lr = 0.0003
    alpha = 0.2
    automatic_entropy_tuning = True
    seed = 42
    batch_size = 256
    num_steps = 1000000
    hidden_size = 256
    updates_per_step = 1
    start_steps = 10000
    target_update_interval = 1
    replay_size = 1000000
    cuda = False


args = Args()

# get the dimensionality of the observation_spec after flattening
flat_obs = tree.flatten(env.observation_spec())
# combine all the shapes
obs_dim = sum([item.shape[0] for item in flat_obs])

# setup agent
agent = SAC(obs_dim, env.action_spec(), args)
# load checkpoint
model_path = 'checkpoints/sac_checkpoint_cheetah_123456_10000'
agent.load_checkpoint(model_path, evaluate=True)

# pull out model
model = agent.policy
# setup hook dict
hook_dict = init_hook_dict(model)
# add hooks
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        module.register_forward_hook(recordtodict_hook(name=name, hook_dict=hook_dict))



geom_names = ['ground', 'torso', 'head', 'bthigh', 'bshin', 'bfoot', 'fthigh', 'fshin', 'ffoot']
joint_names = ['bthigh', 'bshin', 'bfoot', 'fthigh', 'fshin', 'ffoot']
actuator_force_names = ['bthigh', 'bshin', 'bfoot', 'fthigh', 'fshin', 'ffoot']

kinematic_dict = get_kinematics(env.physics, geom_names, joint_names, actuator_force_names)
geom_positions = kinematic_dict['geom_positions']
joint_angles = kinematic_dict['joint_angles']
joint_velocities = kinematic_dict['joint_velocities']
actuator_forces = kinematic_dict['actuator_forces']


actuator_force_names = ['bthigh', 'bshin', 'bfoot', 'fthigh', 'fshin', 'ffoot']
env.physics.named.data.actuator_force

xpos_names = ['ground', 'torso', 'head', 'bthigh', 'bshin', 'bfoot', 'fthigh', 'fshin', 'ffoot']
env.physics.named.data.geom_xpos  # xpos: in world coordinates

qpos_names = ['bthigh', 'bshin', 'bfoot', 'fthigh', 'fshin', 'ffoot']
env.physics.named.data.qpos  # angles

qvel_names = ['bthigh', 'bshin', 'bfoot', 'fthigh', 'fshin', 'ffoot']
env.physics.named.data.qvel  # angle velocity

# run a few episodes just to collect activations
num_episodes_to_run = 10

# for recording kinematics
total_kinematic_dict = {
    'geom_positions': [],
    'joint_angles': [],
    'joint_velocities': [],
    'actuator_forces': [],
    'ffoot_ground': [],
}

for i in range(num_episodes_to_run):
    time_step = env.reset()
    episode_reward = 0

    # note: we don't record kinematics on the reset step

    while not time_step.last():  # or env.get_termination()
        # get the state
        state = get_flat_obs(time_step)
        # sample an action
        action = agent.select_action(state)
        time_step = env.step(action)

        # record kinematics
        kinematic_dict = get_kinematics(env.physics, geom_names, joint_names, actuator_force_names)
        total_kinematic_dict['geom_positions'].append(kinematic_dict['geom_positions'])
        total_kinematic_dict['joint_angles'].append(kinematic_dict['joint_angles'])
        total_kinematic_dict['joint_velocities'].append(kinematic_dict['joint_velocities'])
        total_kinematic_dict['actuator_forces'].append(kinematic_dict['actuator_forces'])

        ffoot_grounded = is_on_ground(env, geom1_name='ground', geom2_name=set(('ffoot',)))
        total_kinematic_dict['ffoot_ground'].append(ffoot_grounded)

        # record reward
        episode_reward += time_step.reward

    print('Episode: {} Reward: {}'.format(i, episode_reward))

# convert the kinematics to numpy arrays
total_kinematic_dict['geom_positions'] = np.stack(total_kinematic_dict['geom_positions'],
                                                  axis=0)  # combine the geom_positions_arr into (t, n, 3)
total_kinematic_dict['joint_angles'] = np.array(total_kinematic_dict['joint_angles'])
total_kinematic_dict['joint_velocities'] = np.array(total_kinematic_dict['joint_velocities'])
total_kinematic_dict['actuator_forces'] = np.array(total_kinematic_dict['actuator_forces'])
total_kinematic_dict['ffoot_ground'] = np.array(total_kinematic_dict['ffoot_ground'])



# get the mapping of the joint names to the joint angles in total_kinematic_dict['joint_angles']
joint_names_to_idx = {joint_name: idx for idx, joint_name in enumerate(joint_names)}

# get the mapping of the actuator force names to the actuator forces in total_kinematic_dict['actuator_forces']
actuator_force_names_to_idx = {actuator_force_name: idx for idx, actuator_force_name in enumerate(actuator_force_names)}

# get the mapping of the geom names to the geom positions in total_kinematic_dict['geom_positions']
geom_names_to_idx = {geom_name: idx for idx, geom_name in enumerate(geom_names)}

idx_to_joint_names = {idx: joint_name for joint_name, idx in joint_names_to_idx.items()}
idx_to_actuator_force_names = {idx: actuator_force_name for actuator_force_name, idx in actuator_force_names_to_idx.items()}
idx_to_geom_names = {idx: geom_name for geom_name, idx in geom_names_to_idx.items()}

# exampl 1: get the joint angles for the bthigh
bthigh_joint_angles = total_kinematic_dict['joint_angles'][:, joint_names_to_idx['bthigh']]
bthigh_joint_angles.shape

# example 2: get the geom positions for the torso
torso_geom_positions = total_kinematic_dict['geom_positions'][:, geom_names_to_idx['torso'], :]
torso_geom_positions.shape


# otherwise, just compile the hook_dict
loaded_hook_dict = compile_hook_dict(hook_dict)

# do pca on the activations
pca_dict = {}
for name, arr in loaded_hook_dict.items():
    pca = PCA(n_components=2)
    pca.fit(arr)
    pca_dict[name] = pca
# %matplotlib inline  # uncomment if you want to see the plots inline
# plot the activations, and save the activations to a dictionary
activations_dict = {}
for name, pca in pca_dict.items():
    # get activations
    activations = get_activations(pca_dict=pca_dict, compiled_hook_dict=loaded_hook_dict, layer_name=name)

    # save activations
    activations_dict[name] = activations

    # plot activations
    save_path = 'activations/{}.png'.format(name)
    # save_path = None
    fig_im = plot_activations(activations, layer_name=name, save_path=save_path, show=True)

for name, activations in loaded_hook_dict.items():
    print(name, activations.shape)

# filter the activations_dict to only include the layers we want
layers_to_include = ['linear1', 'linear2']

# get the activations for the layers we want
filtered_activations = {k: v for k, v in loaded_hook_dict.items() if k in layers_to_include}

# split the activations by the number of episodes collected (assumes all episodes have the same number of steps)
# get the number of steps per episode
num_steps_per_episode = activations_dict['linear1'].shape[0] // num_episodes_to_run

# split the activations by episode
activations_by_episode = {}
for name, activations in activations_dict.items():
    activations_by_episode[name] = activations.reshape(num_episodes_to_run, num_steps_per_episode, -1)

activations_by_episode['linear1'].shape


# get the power spectral density of the pca of the activations
k = 10
psd_pca_dict = {}
psd_avg_dict = {}
for name, activations in filtered_activations.items():
    # get the pca of the activations to k dimensions
    pca = PCA(n_components=k)
    pca.fit(activations)
    activations_pca = pca.transform(activations)

    # get the power spectral density of the pca of the activations
    psd = np.abs(np.fft.fft(activations_pca)) ** 2
    psd_pca_dict[name] = psd

    # average the power spectral density across the k dimensions, weighted by the variance of the pca
    psd_avg = np.average(psd, axis=1, weights=pca.explained_variance_ratio_)
    psd_avg_dict[name] = psd_avg

    # make a plot of the power spectral density
    fig, ax = plt.subplots()
    ax.plot(psd_avg)

    # scale the y axis to be the variance divided by the Hz
    ax.set_ylim(0, np.max(psd_avg))
    ax.set_ylabel('Variance / Hz')
    ax.set_xlabel('Hz')
    ax.set_title('Power Spectral Density of {}'.format(name))

    ax.set_title(name)
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Power')

    save_path = 'psd/{}.png'.format(name)
    # save_path = None
    fig_im = fig2img(fig)
    if save_path is not None:
        fig_im.save(save_path)

f, psd = scipy.signal.welch(total_kinematic_dict['joint_angles'][:1000,:], fs=50, nperseg=500, noverlap=250, axis=-2)

# make a plot of the power spectral density
fig, ax = plt.subplots()
ax.semilogy(f, psd)
ax.set_title('aaaa')
ax.set_xlabel('frequency [Hz]')
ax.set_ylabel('PSD [V**2/Hz]')
# set x scale to log
ax.set_xscale('log')

# get legend labels
labels = ['joint {}'.format(idx_to_joint_names[i]) for i in range(psd.shape[-1])]
ax.legend(labels)

save_path = 'psd/{}.png'.format('aaaa')
# save_path = None
fig_im = fig2img(fig)
if save_path is not None:
    fig_im.save(save_path)

# get the power spectral density of the pca of the activations
k = 10
psd_trajectory_dict = {}
for name, activations in filtered_activations.items():
    # get the pca of the activations to k dimensions
    pca = PCA(n_components=k)
    pca.fit(activations)
    activations_pca = pca.transform(activations)  # (num_episodes_to_run * num_steps_per_episode, k)

    activations_reshaped = activations_pca.reshape(num_episodes_to_run, num_steps_per_episode,
                                                   -1)  # (episodes, steps, k)
    f, psd = scipy.signal.welch(activations_reshaped, fs=50, nperseg=500, noverlap=250, axis=-2)

    psd_pca = np.average(psd, axis=-1, weights=pca.explained_variance_ratio_)  # (episodes, steps)
    psd_trajectory = np.mean(psd_pca, axis=0)  # (steps,)

    psd_trajectory_dict[name] = psd_trajectory

    # make a plot of the power spectral density
    fig, ax = plt.subplots()
    ax.semilogy(f, psd_trajectory)
    ax.set_title(name)
    ax.set_xlabel('frequency [Hz]')
    ax.set_ylabel('PSD [V**2/Hz]')
    # set x scale to log
    ax.set_xscale('log')

    save_path = 'psd/{}.png'.format(name)
    # save_path = None
    fig_im = fig2img(fig)
    if save_path is not None:
        fig_im.save(save_path)

# make a plot of the power spectral density of all the layers from psd_trajectory_dict
fig, ax = plt.subplots()
for name, psd_trajectory in psd_trajectory_dict.items():
    ax.semilogy(f, psd_trajectory, label=name)
ax.set_title('All layers')
ax.set_xlabel('frequency [Hz]')
ax.set_ylabel('PSD [V**2/Hz]')
# set x scale to log
ax.set_xscale('log')
ax.legend()

save_path = 'psd/all_layers.png'
# save_path = None
fig_im = fig2img(fig)
if save_path is not None:
    fig_im.save(save_path)


# event_key = 'ffoot_ground'
# layer_of_interest = 'linear1'

for event_key in ['ffoot_ground']:
    for layer_of_interest in ['linear1', 'linear2']:

        # capture the first bunch of trues in total_kinematic_dict['ffoot_ground']
        # np.indices(total_kinematic_dict['ffoot_ground'] == True).shape

        # get indices where the foot is on the ground
        ffoot_ground_indices = np.where(total_kinematic_dict[event_key] == True)[0]

        # grab a random index from the indices where the foot is on the ground
        random_index = np.random.choice(np.arange(len(ffoot_ground_indices)))
        # make an array of 5 indices around the random index
        contact_indices = np.arange(random_index - 2, random_index + 3)

        indices_to_plot = np.arange(ffoot_ground_indices[contact_indices[0]], ffoot_ground_indices[contact_indices[-1] + 1])
        print(indices_to_plot)

        # get the activations for the indices to plot
        activations_to_plot = loaded_hook_dict[layer_of_interest][ffoot_ground_indices[contact_indices[0]]:ffoot_ground_indices[contact_indices[-1] + 1], :]
        # get the joint angles for the indices to plot
        joint_angles_to_plot = total_kinematic_dict['joint_angles'][ffoot_ground_indices[contact_indices[0]]:ffoot_ground_indices[contact_indices[-1] + 1], :]
        # get the geom positions for the indices to plot
        geom_positions_to_plot = total_kinematic_dict['geom_positions'][ffoot_ground_indices[contact_indices[0]]:ffoot_ground_indices[contact_indices[-1] + 1], :, :]

        # get rid of indices that are clumped together
        # get the indices where the foot is on the ground for more than 1 step
        ffoot_ground_indices = np.where(total_kinematic_dict[event_key] == True)[0]

        final_ffoot_ground_indices = []

        i = 0
        while i < len(ffoot_ground_indices) - 1:
            final_ffoot_ground_indices.append(ffoot_ground_indices[i])
            while ffoot_ground_indices[i + 1] - ffoot_ground_indices[i] == 1:
                i += 1
                if i == len(ffoot_ground_indices) - 1:
                    break
            i += 1


        total_spike_arr = []

        # forr each index in ffoot_ground_indices, grab a neighborhood of 50 indices in both directions
        for index in final_ffoot_ground_indices:
            neighborhood = np.arange(index - 25, index + 25)
            # if the neighborhood is all in ffoot_ground_indices, then we have a good index
            if not np.all(np.isin(neighborhood, np.arange(0, 1000))):
                # modify the neighborhood to only include the indices that are in np.arange(0, 1000)
                # neighborhood = neighborhood[np.isin(neighborhood, np.arange(0, 1000))]

                continue

            spike_neighborhood = loaded_hook_dict[layer_of_interest][neighborhood, :]

            # now save the neighborhood to an array
            total_spike_arr.append(spike_neighborhood)

        total_spike_arr = np.array(total_spike_arr)
        total_spike_arr.shape

        # apply relu on the total_spike_arr
        total_spike_arr_relu = np.maximum(total_spike_arr, 0)

        a = total_spike_arr_relu[15, :, :]

        sorted_activations = a[:, np.argsort(np.argmax(a, axis=0))]

        # make a plot of the sorted activations
        fig, ax = plt.subplots()
        ax.imshow(sorted_activations.T, cmap='magma')

        # place a dotted line at the 25th index, marking the index of the foot contact
        ax.axvline(x=25, color='white', linestyle='--', alpha=0.5)


        # set the x and y labels
        ax.set_xlabel('Time (1/50th sec)')
        ax.set_ylabel('Neuron')

        # set the title
        ax.set_title('Neuron Spiking, {} Layer'.format(layer_of_interest))

        # possible cmap options: 'gray', 'viridis', 'plasma', 'inferno', 'magma', 'cividis''

        # save the figure
        save_path = 'spike_plots/{}.png'.format(layer_of_interest)
        # save_path = None
        fig_im = fig2img(fig)
        if save_path is not None:
            fig_im.save(save_path)