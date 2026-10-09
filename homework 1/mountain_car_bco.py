import gymnasium as gym
import argparse
import pygame
from teleop import collect_demos
import torch
from torch.optim import Adam
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


device = torch.device('cpu')


# This will be useful for implementing BCO
def collect_random_interaction_data(num_iters):
    states = []
    next_states = []
    actions = []

    env = gym.make('MountainCar-v0')

    for i in range(num_iters):
        obs, _ = env.reset()    
        done = False
        while not done:
            a = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(a)
            done = terminated or truncated
            states.append(obs)
            next_states.append(next_obs)
            actions.append(a)
            obs = next_obs
    
    env.close()

    return np.array(states), np.array(next_states), np.array(actions)


def collect_human_demos(num_demos):
    mapping = {(pygame.K_LEFT,): 0, (pygame.K_RIGHT,): 2}
    env = gym.make("MountainCar-v0",render_mode='rgb_array') 
    demos = collect_demos(env, keys_to_action=mapping, num_demos=num_demos, noop=1)
    return demos


def torchify_demos(sas_pairs):
    states = []
    actions = []
    next_states = []
    for s,a, s2 in sas_pairs:
        states.append(s)
        actions.append(a)
        next_states.append(s2)

    states = np.array(states)
    actions = np.array(actions)
    next_states = np.array(next_states)

    obs_torch = torch.from_numpy(np.array(states)).float().to(device)
    obs2_torch = torch.from_numpy(np.array(next_states)).float().to(device)
    acs_torch = torch.from_numpy(np.array(actions)).long().to(device)

    return obs_torch, acs_torch, obs2_torch


def train_policy(obs, acs, nn_policy, num_train_iters):
    pi_optimizer = Adam(nn_policy.parameters(), lr=0.1)
    #action space is discrete so our policy just needs to classify which action to take
    #we typically train classifiers using a cross entropy loss
    loss_criterion = nn.CrossEntropyLoss()
    
    # run BC using all the demos in one giant batch
    for i in range(num_train_iters):
        #zero out automatic differentiation from last time
        pi_optimizer.zero_grad()
        #run each state in batch through policy to get predicted logits for classifying action
        pred_action_logits = nn_policy(obs)
        #now compute loss by comparing what the policy thinks it should do with what the demonstrator didd
        loss = loss_criterion(pred_action_logits, acs) 
        print("iteration", i, "bc loss", loss)
        #back propagate the error through the network to figure out how update it to prefer demonstrator actions
        loss.backward()
        #perform update on policy parameters
        pi_optimizer.step()



class PolicyNetwork(nn.Module):
    '''
        Simple neural network with two layers that maps a 2-d state to a prediction
        over which of the three discrete actions should be taken.
        The three outputs corresponding to the logits for a 3-way classification problem.

    '''
    def __init__(self):
        super().__init__()

        #This layer has 2 inputs corresponding to car position and velocity
        self.fc1 = nn.Linear(2, 8)  
        #This layer has three outputs corresponding to each of the three discrete actions
        self.fc2 = nn.Linear(8, 3)  

    def forward(self, x):
        #this method performs a forward pass through the network, applying a non-linearity (ReLU) on the 
        #outputs of the first layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

class InverseDynamicsNetwork(nn.Module):
    '''
        Predicts which action was taken, given a state and the next state.
        Input is 2-d: position and change in velocity.
        Output is 3 logits, one per action.
    '''
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def make_features(obs, next_obs):
    # an action changes the velocity by a tiny amount, and how much depends on the position (gravity),
    # so use the position and the change in velocity
    return torch.cat([obs[:, :1], next_obs[:, 1:] - obs[:, 1:]], dim=1)


def collect_held_random_data(num_episodes, max_hold=30):
    # like collect_random_interaction_data, but each random action is repeated for 1 to max_hold steps.
    # Changing the action every step just jiggles the car near the bottom of the valley, so the
    # model never sees the positions the demonstrations reach.
    states = []
    next_states = []
    actions = []

    env = gym.make('MountainCar-v0')

    for i in range(num_episodes):
        obs, _ = env.reset()
        done = False
        steps_left = 0
        while not done:
            if steps_left == 0:
                a = env.action_space.sample()
                steps_left = np.random.randint(1, max_hold + 1)
            next_obs, reward, terminated, truncated, info = env.step(a)
            done = terminated or truncated
            states.append(obs)
            next_states.append(next_obs)
            actions.append(a)
            obs = next_obs
            steps_left -= 1

    env.close()

    return np.array(states), np.array(next_states), np.array(actions)


def inverse_dynamics(obs, next_obs, num_iters, ground_truth_acts=None, num_random_episodes=50):
    # train on random interaction data only, never on the demonstrations
    rand_s, rand_s2, rand_a = collect_held_random_data(num_random_episodes)
    rand_x = make_features(torch.from_numpy(rand_s).float(), torch.from_numpy(rand_s2).float())
    rand_a = torch.from_numpy(rand_a).long()

    # scale each input feature to roughly mean 0, std 1 (velocity changes are about 1000x smaller than position)
    mean = rand_x.mean(dim=0)
    std = rand_x.std(dim=0)
    rand_x = (rand_x - mean) / std

    model = InverseDynamicsNetwork()
    optimizer = Adam(model.parameters(), lr=0.01)
    loss_criterion = nn.CrossEntropyLoss()

    for i in range(num_iters):
        optimizer.zero_grad()
        loss = loss_criterion(model(rand_x), rand_a)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print("iteration", i, "inverse dynamics loss", loss.item())

    # estimate the action for every (state, next state) pair in the demonstrations
    with torch.no_grad():
        demo_x = (make_features(obs, next_obs) - mean) / std
        estimated_acts = torch.argmax(model(demo_x), dim=1)
        rand_acc = (torch.argmax(model(rand_x), dim=1) == rand_a).float().mean().item()

    print("inverse dynamics accuracy on random data:", rand_acc)
    if ground_truth_acts is not None:
        demo_acc = (estimated_acts == ground_truth_acts).float().mean().item()
        print("inverse dynamics accuracy on demonstration data:", demo_acc)

    return estimated_acts


#evaluate learned policy
def evaluate_policy(pi, num_evals, human_render=True):
    if human_render:
        env = gym.make("MountainCar-v0",render_mode='human') 
    else:
        env = gym.make("MountainCar-v0") 

    policy_returns = []
    for i in range(num_evals):
        done = False
        total_reward = 0
        obs, _ = env.reset()
        while not done:
            #take the action that the network assigns the highest logit value to
            #Note that first we convert from numpy to tensor and then we get the value of the 
            #argmax using .item() and feed that into the environment
            action = torch.argmax(pi(torch.from_numpy(obs).unsqueeze(0))).item()
            # print(action)
            obs, rew, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += rew
        print("reward for evaluation", i, total_reward)
        policy_returns.append(total_reward)

    print("average policy return", np.mean(policy_returns))
    print("min policy return", np.min(policy_returns))
    print("max policy return", np.max(policy_returns))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--num_demos', default = 1, type=int, help="number of human demonstrations to collect")
    parser.add_argument('--num_bc_iters', default = 100, type=int, help="number of iterations to run BC")
    parser.add_argument('--num_inv_dyn_iters', default = 1000, type=int, help="number of iterations to train inverse dynamics model")
    parser.add_argument('--num_evals', default=6, type=int, help="number of times to run policy after training for evaluation")

    args = parser.parse_args()

    #collect human demos
    demos = collect_human_demos(args.num_demos)

    #process demos
    obs, ground_truth_acts, next_obs = torchify_demos(demos)

    # train the inverse dynamics model on random data and use it to estimate the demo actions
    # (the ground truth actions are only passed in to report accuracy, not used for training)
    estimated_acts = inverse_dynamics(obs, next_obs, args.num_inv_dyn_iters, ground_truth_acts)

    #train policy WITHOUT ground truth actions
    pi = PolicyNetwork()
    train_policy(obs, estimated_acts, pi, args.num_bc_iters)

    #evaluate learned policy
    evaluate_policy(pi, args.num_evals)
