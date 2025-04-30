import numpy as np
from typing import List, Tuple, Any
import random


class SumTree:
    """
    SumTree data structure for efficient sampling from prioritized experience replay
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write_idx = 0
        self.size = 0
        self.max_priority = 1.0
    
    def _propagate(self, idx: int, change: float):
        """Propagate priority update up the tree"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx: int, s: float) -> int:
        """
        Find sample index on the tree with priority closest to s
        """
        left = 2 * idx + 1
        right = left + 1
        
        # If leaf node, return it
        if left >= len(self.tree):
            return idx
        
        # Otherwise, continue down the tree
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self) -> float:
        """Return the total priority"""
        return self.tree[0]
    
    def add(self, data: Any, priority: float):
        """Add a new experience with priority"""
        # Ensure minimum priority is never zero
        priority = max(priority, 1e-5)
        
        # Update max priority
        self.max_priority = max(self.max_priority, priority)
        
        # Store data and update tree
        idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = data
        self.update(idx, priority)
        
        # Update write index
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def update(self, idx: int, priority: float):
        """Update priority of a node"""
        # Ensure minimum priority is never zero
        priority = max(priority, 1e-5)
        
        # Calculate change in priority
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        
        # Propagate change up the tree
        self._propagate(idx, change)
    
    def get(self, s: float) -> Tuple[int, Any, float]:
        """
        Get experience from the tree using priority s
        Returns: (index, data, priority)
        """
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        
        return idx, self.data[data_idx], self.tree[idx]


class PrioritizedReplayBuffer:
    """
    Prioritized Replay Buffer for MAPF training
    """
    def __init__(self, capacity: int, alpha: float = 0.6):
        """
        Initialize a PrioritizedReplayBuffer
        
        Args:
            capacity: Maximum buffer size
            alpha: Priority exponent (0 = uniform, 1 = fully prioritized)
        """
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.capacity = capacity
        self.e = 1e-5  # Small constant to prevent zero priority
    
    def add(self, experience: Tuple, priority: float = None):
        """
        Add an experience to the buffer
        
        Args:
            experience: (state, action, reward, next_state, done)
            priority: Priority value (if None, use max priority)
        """
        # If priority not provided, use max priority
        if priority is None:
            priority = self.tree.max_priority
        
        # Apply alpha for more prioritization
        priority = (priority + self.e) ** self.alpha
        
        # Add to buffer
        self.tree.add(experience, priority)
    
    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List[int], List[Tuple], List[float]]:
        """
        Sample a batch of experiences based on their priorities
        
        Args:
            batch_size: Number of experiences to sample
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)
        
        Returns:
            indices: List of indices in the tree
            batch: List of experiences
            weights: Importance sampling weights
        """
        indices = []
        batch = []
        weights = []
        
        # Calculate segment size
        segment = self.tree.total() / batch_size
        
        # Beta annealing (increases over time for stability)
        beta = min(1.0, beta)
        
        # For calculating max weight
        min_prob = self.tree.tree.min() / self.tree.total()
        max_weight = (min_prob * self.tree.size) ** (-beta)
        
        for i in range(batch_size):
            # Sample from each segment
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            
            # Get experience
            idx, experience, priority = self.tree.get(s)
            
            # Calculate sampling weight
            sampling_prob = priority / self.tree.total()
            weight = (sampling_prob * self.tree.size) ** (-beta)
            # Normalize weights
            weight = weight / max_weight
            
            indices.append(idx)
            batch.append(experience)
            weights.append(weight)
        
        return indices, batch, weights
    
    def update_priorities(self, indices: List[int], priorities: List[float]):
        """
        Update priorities for indices
        
        Args:
            indices: List of indices to update
            priorities: New priorities for those indices
        """
        for idx, priority in zip(indices, priorities):
            priority = (priority + self.e) ** self.alpha
            self.tree.update(idx, priority)
    
    def __len__(self) -> int:
        """Return the current size of the buffer"""
        return self.tree.size