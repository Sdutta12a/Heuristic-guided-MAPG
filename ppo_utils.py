import numpy as np
import torch
from typing import List, Dict, Tuple, Any


def compute_gae(rewards: List[float], values: List[float], dones: List[bool], 
            next_value: float, gamma: float = 0.99, gae_lambda: float = 0.95) -> Tuple[List[float], List[float]]:
    """
    Compute Generalized Advantage Estimation
    
    Args:
        rewards: List of rewards for each step
        values: List of value estimates for each step
        dones: List of done flags for each step
        next_value: Value estimate for the step after the last one
        gamma: Discount factor
        gae_lambda: GAE smoothing parameter
        
    Returns:
        returns: List of discounted returns
        advantages: List of advantage estimates
    """
    advantages = []
    returns = []
    gae = 0
    
    # Convert to numpy arrays for easier operations
    rewards = np.array(rewards)
    values = np.array(values)
    dones = np.array(dones)
    
    # Append next_value to values for calculations
    values = np.append(values, next_value)
    
    # Calculate advantages
    for t in reversed(range(len(rewards))):
        # If it's done, next state value is zero
        if dones[t]:
            next_non_terminal = 0
        else:
            next_non_terminal = 1
            
        delta = rewards[t] + gamma * values[t + 1] * next_non_terminal - values[t]
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages.insert(0, gae)
    
    # Calculate returns
    returns = advantages + values[:-1]
    
    return returns.tolist(), advantages.tolist()


def compute_td_errors(batch: List[Tuple], gamma: float = 0.99) -> List[float]:
    """
    Compute TD errors for prioritized replay
    
    Args:
        batch: List of experiences (state, action, reward, next_state, done)
        gamma: Discount factor
        
    Returns:
        td_errors: List of TD errors
    """
    td_errors = []
    
    for state, action, reward, next_state, done, old_val in batch:
        # Calculate target
        if done:
            target = reward
        else:
            # For simplicity, we're using the old value estimate 
            # In practice, we would use the agent's value network for this
            target = reward + gamma * old_val
        
        # TD error is just the difference between target and old value
        td_error = abs(target - old_val)
        td_errors.append(td_error)
    
    return td_errors


def compute_beta(episode: int, total_episodes: int = 10000, beta_start: float = 0.4, beta_end: float = 1.0) -> float:
    """
    Compute beta parameter for prioritized replay importance sampling
    Beta anneals from beta_start to beta_end over training
    
    Args:
        episode: Current episode number
        total_episodes: Total number of episodes
        beta_start: Initial beta value
        beta_end: Final beta value
        
    Returns:
        beta: Current beta value
    """
    fraction = min(episode / total_episodes, 1.0)
    beta = beta_start + fraction * (beta_end - beta_start)
    return beta


def preprocess_batch(batch: List[Tuple]) -> Tuple[List[Dict], List[int], List[float], List[float], List[float]]:
    """
    Extract and preprocess batch components for PPO training
    
    Args:
        batch: List of (state, action, reward, next_state, done, value) tuples
        
    Returns:
        states: List of state dictionaries
        actions: List of actions
        old_log_probs: List of old log probabilities
        returns: List of returns
        advantages: List of advantages
    """
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []
    values = []
    old_log_probs = []
    
    for experience in batch:
        state, action, reward, next_state, done, value, old_log_prob = experience
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_state)
        dones.append(done)
        values.append(value)
        old_log_probs.append(old_log_prob)
    
    # Compute returns and advantages
    if len(batch) > 0:
        next_value = 0  # Assume terminal state for last experience
        returns, advantages = compute_gae(rewards, values, dones, next_value)
    else:
        returns = []
        advantages = []
    
    return states, actions, old_log_probs, returns, advantages


def collect_rollout(env, agents, max_steps=1000):
    """
    Collect rollout data from environment using current agent policies
    
    Args:
        env: Environment to collect data from
        agents: List of agent policies
        max_steps: Maximum number of steps to collect
        
    Returns:
        experiences: List of experiences for each agent
        info: Environment info at the end of the rollout
    """
    obs = env.reset()
    done = False
    step = 0
    
    # Initialize experiences lists for each agent
    experiences = [[] for _ in range(len(agents))]
    
    while not done and step < max_steps:
        actions = []
        log_probs = []
        values = []
        
        # Select actions
        for i, agent in enumerate(agents):
            action, log_prob, value = agent.select_action(obs[i])
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)
        
        # Execute actions in environment
        next_obs, rewards, dones, info = env.step(actions)
        
        # Determine if episode is done
        done = all(dones)
        
        # Store experiences for each agent
        for i in range(len(agents)):
            experience = (obs[i], actions[i], rewards[i], next_obs[i], dones[i], values[i], log_probs[i])
            experiences[i].append(experience)
        
        # Update observations
        obs = next_obs
        step += 1
    
    return experiences, info
