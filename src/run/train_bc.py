### Step 5: Train the network

# initialize the network
network = BCNetworkDiscrete(env.observation_space.shape[0], env.action_space.n)

# define the optimizer
optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

# define the loss function
loss_fn = nn.CrossEntropyLoss()

# define the number of epochs
num_epochs = 10

# define the batch size
batch_size = 32

# define the number of batches
num_batches = len(storage.obs) // batch_size

# convert the data to tensors
obs = torch.tensor(storage.obs, dtype=torch.float32).squeeze()

# convert integers into one-hot vectors
action = torch.tensor(storage.action, dtype=torch.long).squeeze()
action = torch.nn.functional.one_hot(action, num_classes=env.action_space.n)
# convert action to float32
action = action.float()
print(action.type())

# train the network
for epoch in range(num_epochs):
    # accumulate loss
    epoch_loss = 0
    for batch in range(num_batches):
        # get the batch
        batch_obs = obs[batch * batch_size: (batch + 1) * batch_size]
        batch_action = action[batch * batch_size: (batch + 1) * batch_size]

        # forward pass
        logits = network(batch_obs)

        # compute the loss
        loss = loss_fn(logits, batch_action)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # accumulate loss
        epoch_loss += loss.item()

    # print the loss
    print("Epoch: {}, Loss: {}".format(epoch, epoch_loss / num_batches))