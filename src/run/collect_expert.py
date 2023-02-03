import pickle
from src.bc_utils import TransitionStorage, evaluate_and_collect

# Load the model


### Step 3: Collect some data on an expert.
### You can use the evaluate_and_collect function you wrote above.

storage = TransitionStorage()
evaluate_and_collect(model, storage, num_episodes=1000)


# pickle the storage
