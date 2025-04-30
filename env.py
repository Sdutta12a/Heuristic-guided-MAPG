import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import List, Tuple, Dict, Any
from planner import GlobalPlanner

class WarehouseMAPFEnv(gym.Env):
    """
    Warehouse Multi-Agent Path Finding Environment
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, width: int, height: int, num_agents: int, 
                obstacle_map: np.ndarray, fov: int, planner: GlobalPlanner):
        """
        Initialize the environment
        
        Args:
            width: Width of the grid world
            height: Height of the grid world
            num_agents: Number of agents in the environment
            obstacle_map: 2D binary array where 1 represents obstacles
            fov: Field of view size (square)
            planner: GlobalPlanner instance for path planning
        """
        self.width = width
        self.height = height
        self.num_agents = num_agents
        self.obstacles = obstacle_map
        self.planner = planner
        self.fov = fov
        self.max_steps = 100  # Default max steps before termination
        
        # Action space: move right, left, up, down, stay
        self.action_space = spaces.Discrete(5)
        
        # Observation space: dictionary-based observation
        self.observation_space = spaces.Dict({
            'local_view': spaces.Box(low=0, high=1, shape=(fov, fov), dtype=np.float32),
            'other_agents': spaces.Box(low=0, high=1, shape=(fov, fov), dtype=np.float32),
            'goal_vec': spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            'wp_vec': spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            'distances': spaces.Box(low=0, high=float('inf'), shape=(3,), dtype=np.float32),
        })
        
        # Environment state
        self.positions = []  # Agent positions
        self.goals = []      # Agent goals
        self.global_paths = []  # Global paths for each agent
        self.current_waypoints = []  # Current waypoint for each agent
        self.steps = 0       # Step counter
        
        # For priority-based conflict resolution
        self.agent_priorities = list(range(num_agents))
    
    def reset(self) -> Dict[int, Dict]:
        """
        Reset the environment with random start/goal positions
        
        Returns:
            Dictionary of observations for each agent
        """
        self.steps = 0
        self.positions = []
        self.goals = []
        self.global_paths = []
        self.current_waypoints = []
        
        # Generate random valid starting positions and goals
        valid_positions = self._get_valid_positions()
        if len(valid_positions) < 2 * self.num_agents:
            raise ValueError("Not enough valid positions for all agents")
            
        positions = np.random.choice(len(valid_positions), 2 * self.num_agents, replace=False)
        
        for i in range(self.num_agents):
            # Set start position
            start_pos = valid_positions[positions[i]]
            self.positions.append(start_pos)
            
            # Set goal position
            goal_pos = valid_positions[positions[i + self.num_agents]]
            self.goals.append(goal_pos)
            
            # Plan global path
            path = self.planner.plan_path(start_pos, goal_pos)
            self.global_paths.append(path)
            
            # Set initial waypoint (first position after start)
            if len(path) > 1:
                self.current_waypoints.append(path[1])
            else:
                self.current_waypoints.append(path[0])  # If path is just one point
        
        # Shuffle agent priorities for fairness
        np.random.shuffle(self.agent_priorities)
        
        # Get observations for all agents
        return self._get_all_observations()
    
    def step(self, actions: List[int]) -> Tuple[Dict[int, Dict], List[float], bool, Dict]:
        """
        Execute one step in the environment
        
        Args:
            actions: List of actions for each agent
            
        Returns:
            observations: Dictionary of observations for each agent
            rewards: List of rewards for each agent
            done: Whether the episode is done
            info: Additional information
        """
        self.steps += 1
        old_positions = self.positions.copy()
        
        # Execute actions
        new_positions = self._execute_actions(actions)
        
        # Detect collisions
        collisions = self._detect_collisions(old_positions, new_positions)
        
        # Resolve conflicts through priority-based negotiation
        predicted_conflicts = self.planner.predict_conflicts(
            [self.global_paths[i] for i in range(self.num_agents)]
        )
        
        if collisions or predicted_conflicts:
            self._resolve_conflicts(new_positions, collisions)
        
        # Update agent positions
        self.positions = new_positions
        
        # Update waypoints
        self._update_waypoints()
        
        # Compute rewards
        rewards = self._compute_rewards(old_positions, self.positions, collisions)
        
        # Get observations
        observations = self._get_all_observations()
        
        # Check if all agents reached their goals
        at_goal = [self.positions[i] == self.goals[i] for i in range(self.num_agents)]
        done = all(at_goal) or self.steps >= self.max_steps
        
        # Additional info
        info = {
            'collisions': collisions,
            'at_goal': at_goal,
            'makespan': self.steps,
            'flowtime': sum([self._manhattan_distance(self.positions[i], self.goals[i]) 
                           for i in range(self.num_agents)]),
            'throughput': sum(at_goal) / self.num_agents,
            'success': all(at_goal) and not collisions
        }
        
        return observations, rewards, done, info
    
    def _get_valid_positions(self) -> List[Tuple[int, int]]:
        """Get all valid positions (non-obstacle cells)"""
        valid_positions = []
        for x in range(self.width):
            for y in range(self.height):
                if self.obstacles[x, y] == 0:  # Not an obstacle
                    valid_positions.append((x, y))
        return valid_positions
    
    def _execute_actions(self, actions: List[int]) -> List[Tuple[int, int]]:
        """Execute actions and return new positions"""
        # Action mapping: 0=right, 1=left, 2=down, 3=up, 4=stay
        dx = [1, -1, 0, 0, 0]
        dy = [0, 0, 1, -1, 0]
        
        new_positions = []
        
        for i, action in enumerate(actions):
            x, y = self.positions[i]
            nx, ny = x + dx[action], y + dy[action]
            
            # Check if new position is valid
            if (0 <= nx < self.width and 
                0 <= ny < self.height and 
                self.obstacles[nx, ny] == 0):
                new_positions.append((nx, ny))
            else:
                # Invalid move, stay in place
                new_positions.append((x, y))
                
        return new_positions
    
    def _detect_collisions(self, old_positions: List[Tuple[int, int]], 
                          new_positions: List[Tuple[int, int]]) -> List[Dict]:
        """
        Detect collisions between agents
        
        Args:
            old_positions: Previous positions of agents
            new_positions: New positions of agents
            
        Returns:
            List of collision events
        """
        collisions = []
        
        # Check for vertex collisions (agents at same position)
        for i in range(self.num_agents):
            for j in range(i+1, self.num_agents):
                if new_positions[i] == new_positions[j]:
                    collisions.append({
                        'type': 'vertex',
                        'agents': (i, j),
                        'position': new_positions[i]
                    })
        
        # Check for edge collisions (agents swap positions)
        for i in range(self.num_agents):
            for j in range(i+1, self.num_agents):
                if (new_positions[i] == old_positions[j] and 
                    new_positions[j] == old_positions[i]):
                    collisions.append({
                        'type': 'edge',
                        'agents': (i, j),
                        'positions': (old_positions[i], old_positions[j])
                    })
        
        return collisions
    
    def _resolve_conflicts(self, new_positions: List[Tuple[int, int]], 
                        collisions: List[Dict]) -> None:
        """
        Resolve conflicts based on agent priorities
        
        Args:
            new_positions: New positions of agents
            collisions: List of collision events
        """
        # Create a set of conflicting agents
        conflicting_agents = set()
        for collision in collisions:
            conflicting_agents.add(collision['agents'][0])
            conflicting_agents.add(collision['agents'][1])
        
        # Sort conflicting agents by priority
        sorted_agents = sorted(list(conflicting_agents), 
                              key=lambda a: self.agent_priorities[a])
        
        # Higher priority agents get to move, others stay in place
        for agent in sorted_agents[1:]:  # Skip the highest priority agent
            new_positions[agent] = self.positions[agent]  # Stay in place
    
    def _update_waypoints(self) -> None:
        """Update waypoints for each agent based on current position"""
        for i in range(self.num_agents):
            # Get current path
            path = self.global_paths[i]
            
            # If path is empty or agent reached goal, keep current waypoint
            if not path or self.positions[i] == self.goals[i]:
                continue
            
            # Find agent's position in path
            try:
                current_idx = path.index(self.positions[i])
                
                # Set next waypoint if available
                if current_idx + 1 < len(path):
                    self.current_waypoints[i] = path[current_idx + 1]
                else:
                    # At end of path, waypoint is goal
                    self.current_waypoints[i] = self.goals[i]
            except ValueError:
                # Agent not on path, replan
                new_path = self.planner.plan_path(self.positions[i], self.goals[i])
                self.global_paths[i] = new_path
                
                # Set waypoint to first step in new path
                if len(new_path) > 1:
                    self.current_waypoints[i] = new_path[1]
                else:
                    self.current_waypoints[i] = new_path[0]
    
    def _compute_rewards(self, old_positions: List[Tuple[int, int]], 
                        new_positions: List[Tuple[int, int]], 
                        collisions: List[Dict]) -> List[float]:
        """
        Compute rewards for each agent
        
        Args:
            old_positions: Previous positions of agents
            new_positions: New positions of agents
            collisions: List of collision events
            
        Returns:
            List of rewards for each agent
        """
        rewards = []
        
        # Get set of agents involved in collisions
        collision_agents = set()
        for collision in collisions:
            collision_agents.add(collision['agents'][0])
            collision_agents.add(collision['agents'][1])
        
        for i in range(self.num_agents):
            reward = 0
            
            # Penalty for collisions
            if i in collision_agents:
                reward -= 10
            
            # Reward for moving closer to goal
            old_dist = self._manhattan_distance(old_positions[i], self.goals[i])
            new_dist = self._manhattan_distance(new_positions[i], self.goals[i])
            
            if new_dist < old_dist:
                reward += 1
            elif new_dist > old_dist:
                reward -= 1
            
            # Reward for moving closer to waypoint
            old_wp_dist = self._manhattan_distance(old_positions[i], self.current_waypoints[i])
            new_wp_dist = self._manhattan_distance(new_positions[i], self.current_waypoints[i])
            
            if new_wp_dist < old_wp_dist:
                reward += 0.5
            elif new_wp_dist > old_wp_dist:
                reward -= 0.5
                
            # Big reward for reaching goal
            if new_positions[i] == self.goals[i]:
                reward += 20
                
            # Small penalty for staying in place
            if old_positions[i] == new_positions[i] and new_positions[i] != self.goals[i]:
                reward -= 0.1
            
            rewards.append(reward)
        
        return rewards
    
    def _get_observation(self, agent_id: int) -> Dict:
        """
        Get observation for a specific agent
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Observation dictionary
        """
        pos = self.positions[agent_id]
        goal = self.goals[agent_id]
        waypoint = self.current_waypoints[agent_id]
        
        # Extract local view (FOV)
        local_view = self._extract_fov(pos)
        
        # Encode positions of other agents
        other_agents = self._encode_neighbors(pos)
        
        # Encode goal and waypoint vectors
        goal_vec = self._normalize_vector(goal[0] - pos[0], goal[1] - pos[1])
        wp_vec = self._normalize_vector(waypoint[0] - pos[0], waypoint[1] - pos[1])
        
        # Distances to waypoint and goal
        distances = np.array([
            self._manhattan_distance(pos, waypoint),
            self._manhattan_distance(pos, goal),
            self._manhattan_distance(waypoint, goal)
        ], dtype=np.float32)
        
        return {
            'local_view': local_view,
            'other_agents': other_agents,
            'goal_vec': np.array(goal_vec, dtype=np.float32),
            'wp_vec': np.array(wp_vec, dtype=np.float32),
            'distances': distances
        }
    
    def _get_all_observations(self) -> Dict[int, Dict]:
        """Get observations for all agents"""
        return {i: self._get_observation(i) for i in range(self.num_agents)}
    
    def _extract_fov(self, pos: Tuple[int, int]) -> np.ndarray:
        """
        Extract local field of view around a position
        
        Args:
            pos: Center position for FOV
            
        Returns:
            2D array representing the FOV
        """
        x, y = pos
        half_fov = self.fov // 2
        
        # Initialize FOV with zeros (empty space)
        fov = np.zeros((self.fov, self.fov), dtype=np.float32)
        
        # Fill FOV with environment data
        for i in range(self.fov):
            for j in range(self.fov):
                # Map FOV coordinates to world coordinates
                world_x = x + (i - half_fov)
                world_y = y + (j - half_fov)
                
                # Check if within bounds
                if 0 <= world_x < self.width and 0 <= world_y < self.height:
                    # Mark obstacles
                    if self.obstacles[world_x, world_y] == 1:
                        fov[i, j] = 1
                else:
                    # Out of bounds is marked as obstacle
                    fov[i, j] = 1
        
        return fov
    
    def _encode_neighbors(self, pos: Tuple[int, int]) -> np.ndarray:
        """
        Encode positions of other agents in the FOV
        
        Args:
            pos: Center position for FOV
            
        Returns:
            2D array marking positions of other agents
        """
        x, y = pos
        half_fov = self.fov // 2
        
        # Initialize with zeros
        agent_map = np.zeros((self.fov, self.fov), dtype=np.float32)
        
        # Mark other agents
        for i, other_pos in enumerate(self.positions):
            if other_pos == pos:  # Skip self
                continue
                
            other_x, other_y = other_pos
            
            # Map to FOV coordinates
            fov_x = other_x - x + half_fov
            fov_y = other_y - y + half_fov
            
            # Check if within FOV
            if 0 <= fov_x < self.fov and 0 <= fov_y < self.fov:
                agent_map[fov_x, fov_y] = 1
        
        return agent_map
    
    def _normalize_vector(self, dx: int, dy: int) -> Tuple[float, float]:
        """
        Normalize a 2D vector to unit length
        
        Args:
            dx: x component
            dy: y component
            
        Returns:
            Normalized (x, y) tuple
        """
        length = max(1e-8, np.sqrt(dx**2 + dy**2))  # Avoid division by zero
        return dx / length, dy / length
    
    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def render(self, mode='human'):
        """Render the environment (placeholder)"""
        if mode != 'human':
            raise NotImplementedError(f"Render mode {mode} not supported")
            
        # Simple ASCII rendering
        grid = np.zeros((self.height, self.width), dtype=str)
        grid[:] = '.'
        
        # Mark obstacles
        for x in range(self.width):
            for y in range(self.height):
                if self.obstacles[x, y] == 1:
                    grid[y, x] = '#'
        
        # Mark goals
        for i, goal in enumerate(self.goals):
            gx, gy = goal
            grid[gy, gx] = 'G' + str(i)
        
        # Mark agent positions
        for i, pos in enumerate(self.positions):
            px, py = pos
            grid[py, px] = 'A' + str(i)
        
        # Print grid
        for row in grid:
            print(''.join(row))
        print()
        
    def set_max_steps(self, max_steps: int) -> None:
        """Set maximum number of steps for an episode"""
        self.max_steps = max_steps