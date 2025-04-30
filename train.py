import gymnasium as gym
import torch
import numpy as np
import multiprocessing
from typing import List, Dict, Tuple, Any
import os
import logging
import time
from collections import defaultdict

from env import WarehouseMAPFEnv
from agent import MAPFAgent
from replay import PrioritizedReplayBuffer
from planner import GlobalPlanner
from curriculum import Curriculum
from ppo_utils import compute_beta, compute_td_errors, preprocess_batch, collect_rollout


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MAPF_Training")


def make_vector_envs(env_class, num_agents, planner, curriculum, num_envs=8):
    """
    Create vectorized environments for parallel rollouts
    
    Args:
        env_class: Environment class to instantiate
        num_agents: Number of agents in each environment
        planner: Global planner for path planning
        curriculum: Curriculum for environment configuration
        num_envs: Number of parallel environments
        
    Returns:
        VecEnv: Vectorized environment
    """
    # We'll use a simple wrapper for parallel environments
    class VectorEnv:
        def __init__(self, env_class, configs, num_envs, num_agents, planner):
            self.envs = []
            for _ in range(num_envs):
                config = configs[0]  # Start with the first curriculum stage
                env = env_class(
                    width=20, height=20,  # Default size
                    num_agents=num_agents,
                    obstacle_map=generate_obstacle_map(20, 20, config['density']),
                    fov=5,  # Field of view for agents
                    planner=planner
                )
                self.envs.append(env)
            
            self.num_envs = num_envs
            self.observation_space = self.envs[0].observation_space
            self.action_space = self.envs[0].action_space
        
        def reset(self):
            """Reset all environments"""
            observations = []
            for env in self.envs:
                obs = env.reset()
                observations.append(obs)
            return observations
        
        def reset_with(self, config):
            """Reset with specific config"""
            # Update obstacle map based on config
            new_obstacle_map = generate_obstacle_map(20, 20, config['density'])
            
            observations = []
            for env in self.envs:
                # Update env config
                env.obstacles = new_obstacle_map
                env.planner.obstacle_map = new_obstacle_map  # Update planner obstacles
                env.max_steps = config['max_steps']
                
                # Reset with new config
                obs = env.reset()
                observations.append(obs)
            return observations
        
        def step(self, actions_list):
            """Step all environments with respective actions"""
            next_observations = []
            rewards_list = []
            dones_list = []
            infos_list = []
            
            for i, env in enumerate(self.envs):
                next_obs, rewards, dones, info = env.step(actions_list[i])
                next_observations.append(next_obs)
                rewards_list.append(rewards)
                dones_list.append(dones)
                infos_list.append(info)
            
            return next_observations, rewards_list, dones_list, infos_list
    
    # Generate configs from curriculum
    configs = [stage for stage in curriculum.stages]
    
    # Create vectorized environment
    vec_env = VectorEnv(env_class, configs, num_envs, num_agents, planner)
    
    return vec_env


def generate_obstacle_map(width, height, density):
    """
    Generate a random obstacle map with given density
    
    Args:
        width: Map width
        height: Map height
        density: Obstacle density (0.0 - 1.0)
        
    Returns:
        obstacle_map: 2D boolean array where True indicates obstacle
    """
    obstacle_map = np.random.random((height, width)) < density
    
    # Ensure borders are obstacles
    obstacle_map[0, :] = True
    obstacle_map[-1, :] = True
    obstacle_map[:, 0] = True
    obstacle_map[:, -1] = True
    
    return obstacle_map


def log_metrics(episode, infos):
    """
    Log training metrics
    
    Args:
        episode: Current episode number
        infos: List of info dictionaries from environments
    """
    # Aggregate metrics across environments
    success_rate = sum(info.get('at_goal_flags', []).count(True) for info in infos) / sum(len(info.get('at_goal_flags', [])) for info in infos)
    collision_rate = sum(len(info.get('collisions', [])) for info in infos) / len(infos)
    makespan = np.mean([info.get('makespan', 0) for info in infos])
    flowtime = np.mean([info.get('flowtime', 0) for info in infos])
    throughput = np.mean([info.get('throughput', 0) for info in infos])
    
    logger.info(f"Episode {episode}")
    logger.info(f"Success Rate: {success_rate:.4f}")
    logger.info(f"Collision Rate: {collision_rate:.4f}")
    logger.info(f"Avg Makespan: {makespan:.2f}")
    logger.info(f"Avg Flowtime: {flowtime:.2f}")
    logger.info(f"Avg Throughput: {throughput:.4f}")
    logger.info("-" * 40)
    
    return success_rate, collision_rate, makespan


def save_agents(agents, save_dir="saved_models"):
    """
    Save agent models
    
    Args:
        agents: List of agent models
        save_dir: Directory to save models
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save each agent
    for i, agent in enumerate(agents):
        save_path = os.path.join(save_dir, f"agent_{i}.pt")
        torch.save(agent.state_dict(), save_path)
    
    logger.info(f"Saved {len(agents)} agents to {save_dir}")


def main_train(num_agents=4, num_episodes=10000, log_interval=10, save_interval=100):
    """
    Main training function
    
    Args:
        num_agents: Number of agents in the environment
        num_episodes: Total number of training episodes
        log_interval: Episodes between logging
        save_interval: Episodes between saving models
    """
    start_time = time.time()
    
    # Create obstacle map (initial)
    obstacle_map = generate_obstacle_map(width=20, height=20, density=0.2)
    
    # Initialize components
    planner = GlobalPlanner(obstacle_map)
    curriculum = Curriculum()
    
    # Create vectorized environments (8 environments for parallel rollout)
    envs = make_vector_envs(WarehouseMAPFEnv, num_agents, planner, curriculum, num_envs=8)
    
    # Initialize agents and replay buffers
    agents = [MAPFAgent(envs.observation_space, envs.action_space.n) for _ in range(num_agents)]
    buffers = [PrioritizedReplayBuffer(capacity=100000, alpha=0.6) for _ in range(num_agents)]
    
    # Initialize metrics tracking
    episode_rewards = [[] for _ in range(num_agents)]
    success_rates = []
    collision_rates = []
    makespans = []
    
    for episode in range(num_episodes):
        # Get curriculum configuration for this episode
        cfg = curriculum.get_env_config(episode)
        envs.reset_with(cfg)
        
        # Collect rollouts from all environments
        all_experiences = [[] for _ in range(num_agents)]
        all_infos = []
        
        # For each environment, collect a full episode
        for env_idx in range(envs.num_envs):
            # Use single environment for rollout collection
            single_env = envs.envs[env_idx]
            experiences, info = collect_rollout(single_env, agents)
            
            # Store experiences and info
            for agent_idx in range(num_agents):
                all_experiences[agent_idx].extend(experiences[agent_idx])
            all_infos.append(info)
        
        # Store experiences in replay buffers with priorities
        for agent_idx in range(num_agents):
            for exp in all_experiences[agent_idx]:
                # Compute TD error for priority
                td_error = compute_td_errors(exp, agents[agent_idx])
                buffers[agent_idx].add(exp, td_error)
        
        # Update agents with sampled batches
        for agent_idx in range(num_agents):
            # Skip update if buffer doesn't have enough samples
            if len(buffers[agent_idx]) < 128:
                continue
            
            # Compute beta for prioritized replay
            beta = compute_beta(episode, num_episodes)
            
            # Sample batch from buffer
            batch, weights, indices = buffers[agent_idx].sample(128, beta)
            
            # Preprocess batch for training
            processed_batch = preprocess_batch(batch)
            
            # Train agent on batch
            loss, td_errors = agents[agent_idx].update(processed_batch, weights)
            
            # Update priorities in buffer
            buffers[agent_idx].update_priorities(indices, td_errors)
        
        # Log metrics
        if episode % log_interval == 0:
            success_rate, collision_rate, makespan = log_metrics(episode, all_infos)
            success_rates.append(success_rate)
            collision_rates.append(collision_rate)
            makespans.append(makespan)
            
            # Track average rewards
            for agent_idx in range(num_agents):
                avg_reward = np.mean([exp[2] for exp in all_experiences[agent_idx]])
                episode_rewards[agent_idx].append(avg_reward)
        
        # Save model checkpoints
        if episode % save_interval == 0:
            save_agents(agents, save_dir=f"saved_models/episode_{episode}")
        
        # Update curriculum based on performance
        if episode % curriculum.update_interval == 0 and episode > 0:
            # Use success rate as curriculum metric
            curriculum.update(success_rates[-1])
    
    # Final save of agents
    save_agents(agents, save_dir="saved_models/final")
    
    # Log training time
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.2f} seconds")
    logger.info(f"Final success rate: {success_rates[-1]:.4f}")
    
    return agents, success_rates, collision_rates, makespans


def evaluate(agents, num_eval_episodes=100):
    """
    Evaluate trained agents
    
    Args:
        agents: List of trained agents
        num_eval_episodes: Number of evaluation episodes
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    # Create test environments with different obstacle densities
    test_densities = [0.1, 0.2, 0.3, 0.4]
    num_agents = len(agents)
    
    metrics = {
        'success_rates': [],
        'collision_rates': [],
        'makespans': [],
        'flowtimes': []
    }
    
    for density in test_densities:
        logger.info(f"Evaluating with obstacle density: {density}")
        
        # Create obstacle map
        obstacle_map = generate_obstacle_map(width=20, height=20, density=density)
        
        # Initialize components
        planner = GlobalPlanner(obstacle_map)
        env = WarehouseMAPFEnv(
            width=20, height=20,
            num_agents=num_agents,
            obstacle_map=obstacle_map,
            fov=5,
            planner=planner,
            max_steps=100  # Fixed for evaluation
        )
        
        # Run evaluation episodes
        episode_successes = []
        episode_collisions = []
        episode_makespans = []
        episode_flowtimes = []
        
        for episode in range(num_eval_episodes):
            obs = env.reset()
            done = False
            step = 0
            
            while not done and step < 100:
                actions = []
                for agent_idx in range(num_agents):
                    # Use agent policy (without exploration)
                    action = agents[agent_idx].act(obs[agent_idx], deterministic=True)
                    actions.append(action)
                
                obs, rewards, dones, info = env.step(actions)
                done = all(dones)
                step += 1
            
            # Record metrics
            all_agents_at_goal = info.get('at_goal_flags', []).count(True) == num_agents
            episode_successes.append(all_agents_at_goal)
            episode_collisions.append(len(info.get('collisions', [])))
            episode_makespans.append(info.get('makespan', 0))
            episode_flowtimes.append(info.get('flowtime', 0))
        
        # Calculate metrics for this density
        success_rate = np.mean(episode_successes)
        collision_rate = np.mean(episode_collisions)
        makespan = np.mean(episode_makespans)
        flowtime = np.mean(episode_flowtimes)
        
        logger.info(f"Success Rate: {success_rate:.4f}")
        logger.info(f"Collision Rate: {collision_rate:.4f}")
        logger.info(f"Avg Makespan: {makespan:.2f}")
        logger.info(f"Avg Flowtime: {flowtime:.2f}")
        
        # Store metrics
        metrics['success_rates'].append(success_rate)
        metrics['collision_rates'].append(collision_rate)
        metrics['makespans'].append(makespan)
        metrics['flowtimes'].append(flowtime)
    
    return metrics


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='MAPF Training')
    parser.add_argument('--num_agents', type=int, default=4, help='Number of agents')
    parser.add_argument('--num_episodes', type=int, default=10000, help='Number of training episodes')
    parser.add_argument('--log_interval', type=int, default=10, help='Episodes between logging')
    parser.add_argument('--save_interval', type=int, default=100, help='Episodes between saving models')
    parser.add_argument('--eval', action='store_true', help='Run evaluation on trained agents')
    parser.add_argument('--load_path', type=str, default=None, help='Path to load trained agents')
    args = parser.parse_args()
    
    # Training
    if not args.eval:
        trained_agents, success_rates, collision_rates, makespans = main_train(
            num_agents=args.num_agents,
            num_episodes=args.num_episodes,
            log_interval=args.log_interval,
            save_interval=args.save_interval
        )
        
        # Plot training curves
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.plot(success_rates)
            plt.title('Success Rate')
            plt.xlabel('Episodes (x{})'.format(args.log_interval))
            plt.ylabel('Rate')
            
            plt.subplot(1, 3, 2)
            plt.plot(collision_rates)
            plt.title('Collision Rate')
            plt.xlabel('Episodes (x{})'.format(args.log_interval))
            plt.ylabel('Rate')
            
            plt.subplot(1, 3, 3)
            plt.plot(makespans)
            plt.title('Makespan')
            plt.xlabel('Episodes (x{})'.format(args.log_interval))
            plt.ylabel('Steps')
            
            plt.tight_layout()
            plt.savefig('training_curves.png')
            plt.close()
            
            logger.info("Training curves saved to training_curves.png")
        except ImportError:
            logger.warning("Matplotlib not available, skipping plot generation")
    
    # Evaluation
    else:
        logger.info("Running evaluation mode")
        
        if args.load_path is None:
            logger.error("No load path specified for evaluation")
            exit(1)
        
        # Load trained agents
        trained_agents = []
        for i in range(args.num_agents):
            agent_path = os.path.join(args.load_path, f"agent_{i}.pt")
            if not os.path.exists(agent_path):
                logger.error(f"Agent model not found at {agent_path}")
                exit(1)
            
            agent = MAPFAgent(None, None)  # Placeholder, will be overwritten
            agent.load_state_dict(torch.load(agent_path))
            agent.eval()
            trained_agents.append(agent)
        
        # Run evaluation
        eval_metrics = evaluate(trained_agents)
        
        # Plot evaluation results
        try:
            import matplotlib.pyplot as plt
            
            densities = [0.1, 0.2, 0.3, 0.4]
            
            plt.figure(figsize=(15, 10))
            
            plt.subplot(2, 2, 1)
            plt.plot(densities, eval_metrics['success_rates'], 'o-')
            plt.title('Success Rate vs Obstacle Density')
            plt.xlabel('Obstacle Density')
            plt.ylabel('Success Rate')
            
            plt.subplot(2, 2, 2)
            plt.plot(densities, eval_metrics['collision_rates'], 'o-')
            plt.title('Collision Rate vs Obstacle Density')
            plt.xlabel('Obstacle Density')
            plt.ylabel('Collision Rate')
            
            plt.subplot(2, 2, 3)
            plt.plot(densities, eval_metrics['makespans'], 'o-')
            plt.title('Makespan vs Obstacle Density')
            plt.xlabel('Obstacle Density')
            plt.ylabel('Makespan')
            
            plt.subplot(2, 2, 4)
            plt.plot(densities, eval_metrics['flowtimes'], 'o-')
            plt.title('Flowtime vs Obstacle Density')
            plt.xlabel('Obstacle Density')
            plt.ylabel('Flowtime')
            
            plt.tight_layout()
            plt.savefig('evaluation_results.png')
            plt.close()
            
            logger.info("Evaluation results saved to evaluation_results.png")
        except ImportError:
            logger.warning("Matplotlib not available, skipping plot generation")
