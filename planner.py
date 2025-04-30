import numpy as np
import heapq
from typing import Tuple, List, Dict, Set, Optional


class GlobalPlanner:
    """
    A* based global path planner with caching for warehouse MAPF.
    """
    def __init__(self, obstacle_map: np.ndarray):
        """
        Initialize the global planner with an obstacle map
        
        Args:
            obstacle_map: 2D binary array where 1 represents obstacles
        """
        self.obstacle_map = obstacle_map
        self.width, self.height = obstacle_map.shape
        self.cache = {}  # for repeated start/goal pairs
        
    def plan_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Plan a path from start to goal using A*
        
        Args:
            start: (x, y) starting position
            goal: (x, y) goal position
            
        Returns:
            List of (x, y) coordinates from start to goal
        """
        # Check cache first
        cache_key = (start, goal)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Run A* search
        path = self._astar(start, goal)
        
        # Cache the result
        self.cache[cache_key] = path
        return path
    
    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Manhattan distance heuristic"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighbor positions (4-connected)"""
        x, y = pos
        neighbors = []
        
        # Directions: right, left, down, up
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            # Check bounds and obstacles
            if (0 <= nx < self.width and 
                0 <= ny < self.height and 
                self.obstacle_map[nx, ny] == 0):
                neighbors.append((nx, ny))
                
        return neighbors
    
    def _astar(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """A* search algorithm implementation"""
        # Initialize data structures
        open_set = []  # Priority queue with (f_score, position)
        heapq.heappush(open_set, (self._heuristic(start, goal), start))
        
        came_from = {}  # Path tracking
        
        # Cost from start to node
        g_score = {start: 0}
        
        # Estimated total cost from start to goal through node
        f_score = {start: self._heuristic(start, goal)}
        
        # Set for fast membership testing
        open_set_hash = {start}
        
        while open_set:
            # Get node with lowest f_score
            _, current = heapq.heappop(open_set)
            open_set_hash.remove(current)
            
            # Found goal
            if current == goal:
                path = self._reconstruct_path(came_from, current)
                return path
            
            # Explore neighbors
            for neighbor in self._get_neighbors(current):
                # Tentative g_score
                tentative_g = g_score.get(current, float('inf')) + 1
                
                # Found better path to neighbor
                if tentative_g < g_score.get(neighbor, float('inf')):
                    # Update path
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(neighbor, goal)
                    f_score[neighbor] = f
                    
                    # Add to open set if not already there
                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f, neighbor))
                        open_set_hash.add(neighbor)
        
        # No path found
        return []
    
    def _reconstruct_path(self, came_from: Dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct path from came_from map"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]  # Reverse to get path from start to goal
    
    def predict_conflicts(self, paths: List[List[Tuple[int, int]]]) -> List[dict]:
        """
        Simple look-ahead collision detector on global paths
        
        Args:
            paths: List of paths, one for each agent
            
        Returns:
            List of conflict events
        """
        conflicts = []
        num_agents = len(paths)
        
        # Check for vertex conflicts (agents at same position)
        for i in range(num_agents):
            for j in range(i+1, num_agents):
                path_i = paths[i]
                path_j = paths[j]
                
                # Get the shorter path length
                min_length = min(len(path_i), len(path_j))
                
                for t in range(min_length):
                    # Check if agents are at the same position
                    if path_i[t] == path_j[t]:
                        conflicts.append({
                            'type': 'vertex',
                            'time': t,
                            'agents': (i, j),
                            'location': path_i[t]
                        })
        
        # Check for edge conflicts (agents swap positions)
        for i in range(num_agents):
            for j in range(i+1, num_agents):
                path_i = paths[i]
                path_j = paths[j]
                
                # Get the shorter path length
                min_length = min(len(path_i), len(path_j)) - 1
                
                for t in range(min_length):
                    # Check if agents swap positions
                    if (path_i[t] == path_j[t+1] and 
                        path_i[t+1] == path_j[t]):
                        conflicts.append({
                            'type': 'edge',
                            'time': t,
                            'agents': (i, j),
                            'locations': (path_i[t], path_i[t+1])
                        })
        
        return conflicts
