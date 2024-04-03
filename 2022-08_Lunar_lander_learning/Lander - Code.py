### IMPORTS ###

# General libraries to be imported
import gym
import torch
import random
import numpy as np
from collections import deque, namedtuple

# Import for the plots
# import matplotlib.pyplot as plt





# I had to insert these lines to solve a problem related to the terminal
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"





### VARIABLES DECLARATIONS ###

# Set some variables for the replay buffer
BUFFER_SIZE = int(1e5)  # Dimension of the total replay buffer
MINIBATCH_SIZE = 64 # Dimension of the batch of tuples randomly chosen for the experience replay

# Set some variables for the Q-learning algorithm
GAMMA = 0.99    # Discount factor
ALPHA = 5e-4  # Learning rate

# Set some variables for the network update and the target soft update
LEARNING_STEP = 5  # How often to update the network
TAU = 1e-3  # Value used for the soft update of the target network

# Set the device on which to allocate the torch.Tensors
device = torch.device("cpu")





### ENVIRONMENT INITIALIZATION ###

# Environment initialization from OpenAi Gym
env = gym.make('LunarLander-v2')

print(env)
# Set the seed for the environment for all the random number generators
env.seed(42)

# We can now print the dimension of the state space and action space to check the initialization went fine
print('State space dimension: ', env.observation_space.shape)
print('Action space dimension: ', env.action_space.n)





### NEURAL NETWORK DEFINITION ###

# We use the torch neural network module ###
class NN(torch.nn.Module):

    # Here we initialize the parameters and define the structure of the network
    def __init__(self, state_size, action_size, seed):

        super(NN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = torch.nn.Linear(state_size, 64)  # Input layer of state size
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, action_size) # Output layer of action size
    

    # Here we define a fuction that maps the states in the actions by running through the neural network using relu activation function
    def forward(self, state):

        x = self.fc1(state)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        x = torch.nn.functional.relu(x)
        return self.fc3(x)





### REPLAY BUFFER DEFINITION ###

# Here we buid a fixed size buffer to store the experience tuples (state, action, reward, next state, done)
class ReplayBuffer:

    # Here we initialize the buffer parameters
    def __init__(self, action_size, buffer_size, batch_size, seed):

        self.action_size = action_size
        self.batch_size = batch_size
        self.seed = random.seed(seed)

        # Here we define the structure of the memory as a double ended queue of buffer_size size
        self.memory = deque(maxlen=buffer_size)

        # Here we define the experience as a namedtuple
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    

    # This function is used to add a new experiece tuple to the replay buffer
    def add(self, state, action, reward, next_state, done):

        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    

    # This function is used to sample a random batch of tuples from the memory 
    def sample(self):

        # Here we get the random batch
        experiences = random.sample(self.memory, k=self.batch_size)

        # Here we save the values of each tuple of the batch
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)


    # This function is just used to see the dimension of the memory
    def __len__(self):
        
        return len(self.memory)





### AGENT DEFINITION ###

# Here we define the agent that will interact with the environment and learn from it
class Agent():

    # Here we initialize the agent object
    def __init__(self, state_size, action_size, seed):

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Create the local and the target neural network using the structure previously defined
        self.nn_local = NN(state_size, action_size, seed).to(device)
        self.nn_target = NN(state_size, action_size, seed).to(device)
        # Select the optimizer for the process (will use Adam optimizer)
        self.optimizer = torch.optim.Adam(self.nn_local.parameters(), lr=ALPHA)

        # Assign the replay memory to the replay buffer previously created
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, MINIBATCH_SIZE, seed)

        # Initialize the timestep for the updates
        self.timestep = 0

    
    # Here we have the soft update function to modify the target network parameters
    def soft_update(self, local_model, target_model, tau):

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


    # Here we update the value parameters given the batch of experience tuples from the replay buffer and the discount factor
    def learn(self, experiences, gamma):
        
        # Extract the wanted values from the experiences
        states, actions, rewards, next_states, dones = experiences

        # Now we want to compute and minimize the loss
        # We extract next maximum estimated value from target network
        q_targets_next = self.nn_target(next_states).detach().max(1)[0].unsqueeze(1)
        # We calculate the target value using Bellman equation
        q_targets = rewards + gamma * q_targets_next * (1 - dones)
        # We calculate the expected value from the local network
        q_expected = self.nn_local(states).gather(1, actions)
        
        # Here we calculate the loss using Mean squared error
        loss = torch.nn.functional.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Now we use the soft upgrade function to update the target network
        self.soft_update(self.nn_local, self.nn_target, TAU)
    

    # Here we manage the steps to sample from the replay buffer
    def step(self, state, action, reward, next_state, done):

        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every LEARNING_STEP time steps.
        self.timestep = (self.timestep + 1) % LEARNING_STEP
        if self.timestep == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > MINIBATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)


    # Here we choose the action based on the current policy given the state (array form) and the epsilon for epsilon-greedy policy
    def act(self, state, epsilon=0.):
        
        # Transform the state in the correct form
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        # Evaluate the neural network
        self.nn_local.eval()

        # Evaluate the actions from the state using the neural network
        with torch.no_grad():
            action_values = self.nn_local(state)

        # Train the neural network
        self.nn_local.train()

        # Implement the epsilon-greedy action selection
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

                         

    

### DEEP Q-LEARNING ###

# Here we define the function that actually applies the DQN algorithm with epsilon-greedy policy on diminishing schedule
def dqn(episodes=2000, max_time=1000, epsilon_0=1.0, epsilon_final=0.01, epsilon_decay=0.995):
    
    # We create a list containing scores from each episode and a score window to print the average score per 100 episodes
    scores = []
    scores_window = deque(maxlen=100)

    # We create a list to save the values of epsilon throughout the episode
    epsilon_list = []

    # We initialize epsilon
    epsilon = epsilon_0

    # We loop the algorithm on each episode
    for i in range(1, episodes + 1):

        state = env.reset() # Reset the environment
        score = 0   # Initialize the score

        # For each episode we apply actions until done or until max_time is reached
        for t in range(max_time):
            # Use the act function of the agent to choose the action
            action = agent.act(state, epsilon)
            # Step the environment
            next_state, reward, done, _ = env.step(action)
            # Step the agent to use the replay buffer
            agent.step(state, action, reward, next_state, done)
            # Update state
            state = next_state
            # Increase the score based on the reward
            score += reward
            # If done break the loop
            if done:
                break
        
        # Save the score
        scores_window.append(score)
        scores.append(score)

        # Save the value of epsilon
        epsilon_list.append(epsilon)

        # Decrease epsilon based on diminishing schedule
        epsilon = max(epsilon_final, epsilon_decay*epsilon)


        # Print the average score at the moment and keep the average every 100 episodes
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i, np.mean(scores_window)), end="")
        if i % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i, np.mean(scores_window)))

        # If the average score in the last 100 episodes is more than 200 break the loop and print the rtesult
        if np.mean(scores_window)>=200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i - 100, np.mean(scores_window)))
            torch.save(agent.nn_local.state_dict(), 'checkpoint.pth')
            break
    
    return scores, epsilon_list





### CALLS ###

# Here we call an object of the agent class 
agent = Agent(state_size=8, action_size=4, seed=42)

# Here we launch the DQN function
scores, epsilon_list = dqn()





### PLOT EPSILON ###

# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.plot(np.arange(len(epsilon_list)), epsilon_list)
# plt.ylabel('Epsilon')
# plt.xlabel('Episode #')
# plt.show()





### PLOT SCORES ###

# Define a function to filter the plot and make it smoother
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

# # Plot the scores
# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.plot(np.arange(len(scores)), scores)
# plt.ylabel('Score')
# plt.xlabel('Episode #')
# plt.show()

# # Plot the scores after a running mean of 10
# mean_scores_10 = running_mean(scores, 10)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.plot(np.arange(len(mean_scores_10)), mean_scores_10)
# plt.ylabel('Score')
# plt.xlabel('Episode #')
# plt.show()

# # Plot the scores after a running mean of 20
# mean_scores_20 = running_mean(scores, 20)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.plot(np.arange(len(mean_scores_20)), mean_scores_20)
# plt.ylabel('Score')
# plt.xlabel('Episode #')
# plt.show()

# # Plot the scores after a running mean of 50
# mean_scores_50 = running_mean(scores, 50)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.plot(np.arange(len(mean_scores_50)), mean_scores_50)
# plt.ylabel('Score')
# plt.xlabel('Episode #')
# plt.show()