# Train Environment and Q-learning Agent

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pygame
import random
random.seed(48)
np.random.seed(48)

class Train:
    def __init__(self, start, goal, name, departure_time, arrival_time):
        self.name = name
        self.start = start
        self.position = start
        self.goal = goal
        self.departure_time = departure_time
        self.arrival_time = arrival_time
        self.actual_departure_time = None
        self.actual_arrival_time = None
        self.reward = 0
        self.crashed = False
        self.done = False

class TrainEnvironment:
    def __init__(self, n_states, trains):
        self.n_states = n_states
        self.trains = trains
        self.stations = {'A': 0, 'B': 4, 'C': 8}  # Define the train stations
        self.railway_switch = {}
        self.total_states = n_states #* len(trains)
        self.window_size = 512
        self.window = None
        self.clock = None
        self.reset()

    def reset(self):
        for train in self.trains:
            train.position = train.start
            train.actual_arrival_time = None
            train.reward = 0
            train.done = False
            train.crashed = False
        return [train.position for train in self.trains]

    def step(self, train_id, action, step):
        train = self.trains[train_id]
     
        if action == 0:  # Move left
            train.position = max(0, train.position - 1)
        elif action == 1:  # Stay
            pass
        elif action == 2:  # Move right
            train.position = min(self.n_states - 1, train.position + 1)
            
        # Check for collisions
        reward = 0
        delta_arrival_time = 0
        positions = [t.position for t in self.trains]
        if positions.count(train.position) > 1:
            if train.position not in self.stations.values() or self.railway_switch:
                for v in self.trains:
                    if v.position == train.position and v.position not in self.stations.values():
                        v.reward = -100  # Collision penalty
                        v.done = True
                        v.crashed = True
            else:
                pass
        # Check if the train has reached its goal
        if train.position == train.goal and train.crashed != True:
            train.reward = 100 # Goal reward
            train.done = True
            # Update actual arrival time if the train reaches its goal
            if train.actual_arrival_time is None:
                train.actual_arrival_time = step
        ## Delay Berechnung
        if train.actual_arrival_time is not None and train.crashed != True:
            delta_arrival_time = train.actual_arrival_time - train.arrival_time
            if delta_arrival_time > 0:
                train.reward -= delta_arrival_time

        reward = train.reward
        return [t.position for t in self.trains], reward, delta_arrival_time
    
    def render_frame(self, env, step, episode, total_reward, actions, render = 3):
        # initialisation of pygame
        if self.window == None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock == None:    
            self.clock = pygame.time.Clock()
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = self.window_size / self.n_states

        # visual basics
        colors = [[163, 28, 6], [189, 70, 62], [57, 35, 188], [26, 173, 189], [228, 139, 22], [151, 108, 8], [7, 23, 55], [59, 129, 154], [6, 143, 50], [183, 166, 179], [139, 107, 56], [114, 150, 71], [207, 222, 1], [194, 206, 40], [178, 108, 87], [71, 39, 55], [245, 195, 86], [26, 23, 97], [24, 91, 216], [88, 154, 67]]
        font = pygame.font.Font('freesansbold.ttf', 20)

        # First we draw the targets
        for j in range(len(self.trains)):
            pygame.draw.rect(
                canvas,
                colors[j],
                pygame.Rect(
                    (pix_square_size * self.trains[j].goal, 0),
                    (pix_square_size, pix_square_size),
                ),
            )
        
        # Now we draw the agent
        circle_int = pix_square_size
        for k in range(len(self.trains)):
            pygame.draw.circle(
                canvas,
                colors[k],
                ((self.trains[k].position + 0.5) * pix_square_size, 0.5 * pix_square_size),
                circle_int / 3,
            )
            canvas.blit(font.render(str(self.trains[k].name), False, (0,0,0)), ((self.trains[k].position + 0.5) * pix_square_size ,0.4 * pix_square_size))
            circle_int -= 5

        # Finally, add some gridlines
        for x in range(self.n_states + 1):
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, pix_square_size * 1),
                width=3,
            )

        for x in range(2):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )

        canvas.blit(font.render('Step: ' + str(step), False, (0,0,0)), (25,325))
        canvas.blit(font.render('Episode: ' + str(episode), False, (0,0,0)), (25,300))
        canvas.blit(font.render('Total Reward: ' + str(total_reward), False, (0,0,0)), (25,350))
        new_actions = []
        
        for i in actions:
            if i[-1] == 0:
                new_actions.append(str(i[0]) + ': left')
            elif i[-1] == 1:
                new_actions.append(str(i[0]) + ': stay')
            elif i[-1] == 2:
                new_actions.append(str(i[0]) + ': right')
            else:
                new_actions.append(str(i[0]) + ': None')
            
        canvas.blit(font.render('Actions: ' + str(new_actions), False, (0,0,0)), (25,375))
        agent_rewards = [i.reward for i in env.trains]
        canvas.blit(font.render('Agent Rewards: ' + str(agent_rewards), False, (0,0,0)), (25,400))
        dones = [(i.name, i.done) for i in env.trains]
        canvas.blit(font.render('Done: ' + str(dones), False, (0,0,0)), (25,425))
        position = [(i.name, i.position) for i in env.trains]
        canvas.blit(font.render('Position: ' + str(position), False, (0,0,0)), (25,450))

        # The following line copies our drawings from `canvas` to the visible window
        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(render)

class QLearningAgent:
    def __init__(self, n_states, n_actions, learning_rate, discount_factor, exploration_rate, exploration_decay):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.q_table = np.zeros((n_states, n_actions))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.n_actions)
        else:
            qtable = self.q_table[state, :]
            if sum(qtable == np.argmax(qtable)) > 1:
                cur_index = np.where(qtable == np.max(qtable))[0]
                action = random.choice(cur_index)
            return np.argmax(self.q_table[state, :])
    
    def choose_action_evaluation(self, state):
        qtable = self.q_table[state, :]
        if sum(qtable == np.argmax(qtable)) > 1:
            cur_index = np.where(qtable == np.max(qtable))[0]
            action = random.choice(cur_index)
        return np.argmax(self.q_table[state, :])

    def learn(self, state, action, reward, next_state):
        predict = self.q_table[state, action]
        target = reward + self.gamma * np.max(self.q_table[next_state, :])
        self.q_table[state, action] = self.q_table[state, action] + self.lr * (target - predict)

def train_agents(env, agents, n_episodes, max_steps, viz_on):
    rewards_per_episode = []
    deltas_arrival = []
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        delay_per_episode = 0
        done = [False] * len(agents)
        step = 0
        crashed = False # Flag to check if the episode should end due to a collision
                
        while not all(done) and step < max_steps:
            curr_action = []
            for i, agent in enumerate(agents):
                if not done[i] and step >= env.trains[i].departure_time:
                    action = agent.choose_action(state[i])
                    curr_action.append((i, action))
                    next_state, reward, delta_arrival_time = env.step(i, action, step)
                    print(f"Train {i} Action: {action}")
                    agent.learn(state[i], action, reward, next_state[i])
                    delay_per_episode += delta_arrival_time
                    print('Reward', reward)
                    if next_state[i] == env.trains[i].goal:
                        done[i] = True
                else:
                    curr_action.append((i, None))
            state = next_state  # Update the state
            print(f"Train State: {state}")
            step += 1
            total_reward = np.sum([i.reward for i in env.trains])
            if viz_on == 1:
                env.render_frame(env, step, episode, total_reward, curr_action)
            if True in [m.crashed for m in env.trains]:  # Break out of the outer loop if the episode should end
                env.reset()
                break    
        # Penalty if agent did not arrive in Goal state
        for i in env.trains:
            if i.actual_arrival_time == None:
                i.actual_arrival_time = max_steps
                delay_per_episode += i.actual_arrival_time - i.arrival_time
           
        rewards_per_episode.append(total_reward)
        deltas_arrival.append(delay_per_episode)
        env.reset()  # Reset the environment at the end of each episode
    pygame.display.quit()
    pygame.quit()
    return rewards_per_episode, deltas_arrival

def evaluate_agent(env, agent, n_eval_episodes, max_steps, q_table, viz_on):
    rewards_per_episode = []
    deltas_arrival = []
    for episode in range(n_eval_episodes):
        state = env.reset()
        total_reward = 0
        delay_per_episode = 0
        done = [False] * len(agents)
        step = 0
        crashed = False  # Flag to check if the episode should end due to a collision
        
        while not all(done) and step < max_steps:
            curr_action = []
            for i, agent in enumerate(agents):
                if not done[i] and step >= env.trains[i].departure_time:
                    action = agent.choose_action_evaluation(state[i])
                    curr_action.append((i, action))
                    next_state, reward, delta_arrival_time = env.step(i, action, step)
                    print(f"Train {i} Action (Evaluation): {action}")
                    delay_per_episode += delta_arrival_time
                    if next_state[i] == env.trains[i].goal:
                        done[i] = True
                else:
                    curr_action.append((i, None))
            state = next_state  # Update the state
            print(f"Train State (Evaluation): {state}")
            step += 1
            total_reward = np.sum([i.reward for i in env.trains])
            if viz_on == 1:
                env.render_frame(env, step, episode, total_reward, curr_action)    
            if True in [m.crashed for m in env.trains]:  # Break out of the outer loop if the episode should end
                env.reset()
                break
         # Penalty if agent did not arrive in Goal state
        for i in env.trains:
            if i.actual_arrival_time == None:
                i.actual_arrival_time = max_steps
                delay_per_episode += i.actual_arrival_time - i.arrival_time

        rewards_per_episode.append(total_reward)
        deltas_arrival.append(delay_per_episode)
        env.reset()  # Reset the environment at the end of each episode
        mean_reward = np.mean(rewards_per_episode)
        std_reward = np.std(rewards_per_episode)
    pygame.display.quit()
    pygame.quit()
    return mean_reward, std_reward, deltas_arrival

def print_q_table(q_table, agent_id):
    print(f"\nAgent {agent_id} Q-Table:")
    print("-" * 40)
    print("State | Action 0 | Action 1 | Action 2")
    print("-" * 40)
    for state, actions in enumerate(q_table):
        print(f"{state:5} | {actions[0]:8.2f} | {actions[1]:8.2f} | {actions[2]:8.2f}")
    print("-" * 40)

def grid_search(env, agents):
    Results = pd.DataFrame(columns = ['training_episodes', 'max_epsilon', 'decay_rate', 'learning_rate', 'gamma', 'mean_reward', 'mean_lateness'])
    n_training_episodes = [15000]
    epsilon = [0.2, 0.4, 0.6]
    decay_rate = [0.95, 0.995]
    max_steps = 300
    learning_rate = [0.9, 0.6, 0.3]
    gamma = [0.8, 0.9]

    for a in n_training_episodes:
        for b in epsilon:
            for c in decay_rate:
                for d in learning_rate:
                    for e in gamma:
                        for f in agents:
                            f.lr = d
                            f.gamma = e
                            f.epsilon = b
                            f.epsilon_decay = c
                        rewards_per_episode, deltas_arrival = train_agents(env, agents, n_episodes = a, max_steps = max_steps, viz_on = 0)
                        dict_frame = {'training_episodes': [a], 'max_epsilon': [b], 'decay_rate': [c], 'learning_rate': [d], 'gamma': [e], 'mean_reward': [np.mean(rewards_per_episode)], 'mean_lateness': [np.mean(deltas_arrival)]}
                        Results = pd.concat([Results, pd.DataFrame.from_dict(dict_frame)], ignore_index = True)
                        Results = Results.sort_values(by='mean_reward', ascending=False)
                        print(Results.iloc[0:10,:])
    return Results
                        

# Initialize 'Planned-Schedule' and agents
trains = [
    Train(0, 4, 0, departure_time=3, arrival_time=7),
    Train(4, 0, 1, departure_time=8, arrival_time=12),
    Train(0, 8, 2, departure_time=0, arrival_time=8),
    Train(8, 4, 3, departure_time=0, arrival_time=4),
    Train(4, 8, 4, departure_time=7, arrival_time=11) 
]
env = TrainEnvironment(9, trains)
agents = [QLearningAgent(env.total_states, 3, learning_rate=0.3, discount_factor=0.8, exploration_rate=0.2, exploration_decay=0.950) for _ in env.trains]

# Train the agents
rewards, delay_per_episode = train_agents(env, agents, n_episodes=15000, max_steps=300, viz_on = 0)

# Performance Evaluation - based on learned q-values
mean_reward, std_reward, delay_per_episode_eval = evaluate_agent(env, agents, n_eval_episodes=1000, max_steps=300, q_table = rewards, viz_on = 0)

# Initialize 'Delayed-Schedule' and agents
trains_delayed = [
    Train(0, 4, 0, departure_time=3, arrival_time=7),
    Train(4, 0, 1, departure_time=8, arrival_time=12),
    Train(0, 8, 2, departure_time=5, arrival_time=8),
    Train(8, 4, 3, departure_time=0, arrival_time=4),
    Train(4, 8, 4, departure_time=7, arrival_time=11)
]
env_delayed = TrainEnvironment(9, trains_delayed)
agents_delayed = [QLearningAgent(env_delayed.total_states, 3, learning_rate=0.3, discount_factor=0.8, exploration_rate=0.2, exploration_decay=0.950) for _ in env_delayed.trains]

# Train the agents - Delayed Schedule
rewards_delayed, delay_per_episode_delayed = train_agents(env_delayed, agents_delayed, n_episodes=15000, max_steps=300, viz_on = 0)

# Performance Evaluation of delayed schedule - Same Q-Values
mean_reward_delayed, std_reward_delayed, delay_per_episode_eval_delayed = evaluate_agent(env_delayed, agents_delayed, n_eval_episodes=1000, max_steps=300, q_table = rewards_delayed, viz_on = 0)


# Output Performance Evaluation - Standard Schedule and Delayed Schedule
print(f"Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
print(f"Mean_reward Delayed ={mean_reward_delayed:.2f} +/- {std_reward_delayed:.2f}")

# Hyperparameter Tuning for Optimal and delayed Model
#hyperparameter_tuning = grid_search(env,agents)
#hyperparameter_tuning_delayed = grid_search(env_delayed,agents_delayed)

# Output Q-Table
for i, agent in enumerate(agents):
    print_q_table(agent.q_table, i)
    
# Output Q-Table 'Delayed-Schedule'
for i, agent in enumerate(agents_delayed):
    print_q_table(agent.q_table, i)

# Visualize the training progress
plt.figure(figsize=(10, 6))
plt.bar(range(len(rewards)),rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Reward per Episode')
plt.grid(True)
plt.show()

# Visualize the training progress
plt.figure(figsize=(10, 6))
plt.bar(range(len(delay_per_episode)),delay_per_episode)
plt.xlabel('Episode')
plt.ylabel('Total Delay')
plt.title('Total Delay per Episode')
plt.grid(True)
plt.show()

# Visualize the training progress
plt.figure(figsize=(10, 6))
plt.bar(range(len(rewards_delayed)),rewards_delayed)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Reward per Episode (Delayed)')
plt.grid(True)
plt.show()

# Visualize the training progress
plt.figure(figsize=(10, 6))
plt.bar(range(len(delay_per_episode_delayed)),delay_per_episode_delayed)
plt.xlabel('Episode')
plt.ylabel('Total Delay')
plt.title('Total Delay per Episode (Delayed)')
plt.grid(True)
plt.show()