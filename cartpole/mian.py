import numpy as np
import time
import gymnasium as gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

env = gym.make('CartPole-v1', render_mode='rgb_array')
obs, info = env.reset(seed=42)

class Simplemodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(4,5), #5 Input and 5 output features
            nn.ReLU(),
            nn.Linear(5,1), #5 input and just one output with sigmoid AF to classify betn take or not take action
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x) 
    
model = Simplemodel()
# print(model)

def play_one_step(env, obs, model, loss_fn):
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    
    left_prob = model(obs_tensor) #probability output kiya rhega betweeen 1 or 0 ---->so keep in mind and that too of going LEFT
    
    action = ( torch.rand(1) > left_prob).float() #typecasting -> here action is either : 0 || 1
    
    """
    if action is ZERO then : Meaning that agent should GOTO LEFT
    the target probability of going left will be --------> y_target => 1 - 0 => 1
    
    if action is ONE then : Meaning that agent should GOTO RIGHT
    the target probability of going right will be -------> y_target => 1 - 1 => 0
    """
    
    # y_target = torch.tensor([[1.]]) - action
    y_target = 1.0 - action
    
    loss = loss_fn( y_target.view_as(left_prob), left_prob)
    
    
    #Computing gradaeint
    model.zero_grad() #Clearing the previous gradients
    loss.backward() # Backpropagation
    
    obs, reward, done, truncated, info= env.step(int(action.item()))
    
    
    #For each action taken, the gradients that would make the action more likely are computed (but not applied yet).
    grads = [param.grad.clone() for param in model.parameters() if param.grad is not None]
    
    return obs, reward, done, truncated, grads

# play_one_step([10, 0, -50, 10, 20], env,nn.BCELoss(),model)

# The above code was for the one step and here this is for multiple episodes
# The code here is self explanatory
def play_multiple_episodes(env, n_episodes, n_max_steps, model, loss_fn):
    all_rewards = []
    all_grads = []
    for episode in range(n_episodes):
        curr_rewards = []
        curr_grads = []
        obs, info = env.reset()
        for step in range(n_max_steps):
            obs, rewards, done, truncated, grads = play_one_step(env,obs,model,loss_fn)
            curr_rewards.append(int(rewards)) #rewards could be 1,1,0,1,0,1,0,0,1,1,0,1
            curr_grads.append(grads)     #grads could be
            if done or truncated:
                break
        all_rewards.append(curr_rewards)
        all_grads.append(curr_grads)
    return all_rewards, all_grads


def discount_rewards(rewards, discount_factor):
    discounted = np.array(rewards)
    for step in range(len(rewards)-2, -1, -1):
        discounted[step] += discounted[step+1] * discount_factor
    return discounted

def discount_and_normalize_rewards(all_rewards, discount_factor):
    all_discounted_rewards = [discount_rewards(reward, discount_factor) for reward in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    mean = flat_rewards.mean()
    std = flat_rewards.std()
    return [(discounted_rewards-mean)/std for discounted_rewards in all_discounted_rewards]
    
    
n_iterations = 150
n_episodes_per_update = 10
n_max_steps = 200
discount_factor = 0.95

optimizer = torch.optim.NAdam(model.parameters(), lr=0.01)
loss_fn = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss
# loss_fn = nn.MSELoss()  # Binary cross-entropy loss


for iteration in range(n_iterations):
    all_rewards, all_grads = play_multiple_episodes(
        env, n_episodes_per_update, n_max_steps, model, loss_fn
    )

    all_final_rewards = discount_and_normalize_rewards(all_rewards, discount_factor)

    all_mean_grads = []
    for var_index, param in enumerate(model.parameters()):
        mean_grads = torch.mean(
            torch.stack([
                final_reward * all_grads[episode_index][step][var_index]
                for episode_index, final_rewards in enumerate(all_final_rewards)
                for step, final_reward in enumerate(final_rewards)
            ]),
            dim=0
        )
        all_mean_grads.append(mean_grads)

    # Apply gradients manually
    for param, mean_grad in zip(model.parameters(), all_mean_grads):
        param.grad = mean_grad  # Set gradients manually

    optimizer.step()  # Update model parameters
    optimizer.zero_grad()  # Clear accumulated gradients

def test_model(env, model, n_episodes=10):
    total_rewards = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        steps = 0
        
        while not (done or truncated):
            # Convert observation to tensor
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            
            # Get action probabilities from the model
            left_prob = model(obs_tensor)
            # Since we're using BCEWithLogitsLoss, we need to apply sigmoid
            left_prob = torch.sigmoid(left_prob)
            
            # Choose action deterministically for testing
            action = 0 if left_prob.item() > 0.5 else 1
            
            # Take action in the environment
            obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            steps += 1
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode+1}: Reward = {episode_reward}, Steps = {steps}")
    
    avg_reward = sum(total_rewards) / len(total_rewards)
    print(f"\nAverage reward over {n_episodes} episodes: {avg_reward:.2f}")
    
    return total_rewards

test_rewards = test_model(env, model, n_episodes=20)


def visualize_model(env, model, n_episodes=3):
    for episode in range(n_episodes):
        obs, info = env.reset()
        env.render()  # Make sure your environment supports rendering
        
        done = False
        truncated = False
        total_reward = 0
        
        while not (done or truncated):
            # Convert observation to tensor
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            
            # Get action probabilities from the model
            left_prob = model(obs_tensor)
            # Apply sigmoid since we're using BCEWithLogitsLoss
            left_prob = torch.sigmoid(left_prob)
            
            # Choose action deterministically for visualization
            action = 0 if left_prob.item() > 0.5 else 1
            
            # Take action in the environment
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            # Render the environment
            env.render()
            time.sleep(0.05)  # Slow down visualization
            
        print(f"Episode {episode+1}: Total reward = {total_reward}")
    
    env.close()
    
visualize_model(env, model)





