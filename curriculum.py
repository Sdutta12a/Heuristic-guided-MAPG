"""
Curriculum Learning module for MAPF environments
This module implements curriculum learning to gradually increase difficulty
"""

import numpy as np
import logging

logger = logging.getLogger("MAPF_Curriculum")


class Curriculum:
    """
    Curriculum learning for MAPF environments.
    
    Manages a progression of environment configurations with increasing difficulty.
    Difficulty is controlled by obstacle density, map size, and maximum steps.
    """
    
    def __init__(self, initial_density=0.1, max_density=0.4, update_interval=100):
        """
        Initialize curriculum learning.
        
        Args:
            initial_density: Initial obstacle density
            max_density: Maximum obstacle density
            update_interval: Episodes between curriculum updates
        """
        self.update_interval = update_interval
        
        # Define curriculum stages with increasing difficulty
        self.stages = [
            # Stage 1: Easy environments
            {
                'density': initial_density,
                'max_steps': 50,
                'min_success_rate': 0.7  # Success rate to advance to next stage
            },
            # Stage 2: Medium difficulty
            {
                'density': initial_density + (max_density - initial_density) * 0.33,
                'max_steps': 75,
                'min_success_rate': 0.6
            },
            # Stage 3: Hard environments
            {
                'density': initial_density + (max_density - initial_density) * 0.67,
                'max_steps': 100,
                'min_success_rate': 0.5
            },
            # Stage 4: Very hard environments
            {
                'density': max_density,
                'max_steps': 125,
                'min_success_rate': 0.0  # Final stage has no advancement criteria
            }
        ]
        
        # Current curriculum stage
        self.current_stage = 0
        
        # Success rate history for stage advancement
        self.success_history = []
        self.history_window = 10  # Number of updates to average for advancement
        
        logger.info(f"Initialized curriculum with {len(self.stages)} stages")
        logger.info(f"Initial stage: density={self.stages[0]['density']}, max_steps={self.stages[0]['max_steps']}")
    
    def get_env_config(self, episode):
        """
        Get environment configuration for the current episode.
        
        Args:
            episode: Current episode number
            
        Returns:
            config: Dictionary with environment configuration
        """
        # Get current stage config
        config = self.stages[self.current_stage].copy()
        
        # Remove advancement criteria from returned config
        if 'min_success_rate' in config:
            del config['min_success_rate']
        
        return config
    
    def update(self, success_rate):
        """
        Update curriculum based on agent performance.
        
        Args:
            success_rate: Current success rate
            
        Returns:
            advanced: Whether curriculum advanced to next stage
        """
        # Store success rate
        self.success_history.append(success_rate)
        
        # Keep history window limited
        if len(self.success_history) > self.history_window:
            self.success_history = self.success_history[-self.history_window:]
        
        # Only update if we have enough history
        if len(self.success_history) < self.history_window:
            return False
        
        # Check if we should advance to next stage
        avg_success = np.mean(self.success_history)
        
        if self.current_stage < len(self.stages) - 1:
            min_success_rate = self.stages[self.current_stage]['min_success_rate']
            
            if avg_success >= min_success_rate:
                # Advance to next stage
                self.current_stage += 1
                
                # Reset success history for new stage
                self.success_history = []
                
                logger.info(f"Advanced to curriculum stage {self.current_stage + 1}")
                logger.info(f"New config: density={self.stages[self.current_stage]['density']}, "
                           f"max_steps={self.stages[self.current_stage]['max_steps']}")
                
                return True
        
        return False
    
    def get_current_stage(self):
        """
        Get current curriculum stage.
        
        Returns:
            stage_idx: Current stage index
            config: Current stage configuration
        """
        return self.current_stage, self.stages[self.current_stage]
    
    def reset(self):
        """Reset curriculum to initial stage"""
        self.current_stage = 0
        self.success_history = []
        
        logger.info("Reset curriculum to initial stage")
