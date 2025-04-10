import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
from collections import deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
import os
import joblib
import json

logger = logging.getLogger(__name__)

class BudgetEnvironment:
    def __init__(self, historical_data: pd.DataFrame, categories: List[str], 
                 income: float, savings_goal: float, time_horizon: int):
        self.categories = categories
        self.income = income
        self.savings_goal = savings_goal
        self.time_horizon = time_horizon
        self.historical_data = historical_data
        self.current_month = 0
        self.current_savings = 0
        self.reset()
        
    def reset(self):
        """Reset the environment to initial state"""
        self.current_month = 0
        self.current_savings = 0
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current state representation"""
        try:
            logger.debug("Preparing state representation")
            # Calculate historical spending patterns
            if self.historical_data.empty:
                logger.warning("Empty historical data, using zeros for state")
                state = np.zeros(len(self.categories) + 2)
                state[-2] = self.income
                state[-1] = self.current_savings / max(self.savings_goal, 1e-6)  # Avoid division by zero
                return state
                
            historical_spending = self.historical_data.groupby('category')['amount'].mean()
            logger.debug(f"Historical spending calculated: {len(historical_spending)} categories")
            
            state = np.zeros(len(self.categories) + 2)  # +2 for income and savings
            
            # Fill in category spending ratios
            for i, cat in enumerate(self.categories):
                spending = historical_spending.get(cat, 0)
                state[i] = spending / max(self.income, 1e-6)  # Avoid division by zero
                
            # Add income and savings goal progress
            state[-2] = self.income
            state[-1] = self.current_savings / max(self.savings_goal, 1e-6)  # Avoid division by zero
            
            logger.debug(f"State prepared with shape: {state.shape}")
            return state
        except Exception as e:
            logger.error(f"Error preparing state: {str(e)}", exc_info=True)
            # Return a safe fallback state
            fallback_state = np.zeros(len(self.categories) + 2)
            fallback_state[-2] = self.income
            return fallback_state
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """
        Execute one step in the environment
        action: budget allocation for each category (sums to 1)
        """
        # Calculate actual spending based on historical patterns
        actual_spending = np.zeros(len(self.categories))
        for i, cat in enumerate(self.categories):
            historical_avg = self.historical_data[self.historical_data['category'] == cat]['amount'].mean()
            actual_spending[i] = min(action[i] * self.income, historical_avg * 1.2)  # Cap at 20% above historical
            
        # Calculate savings
        total_spent = np.sum(actual_spending)
        monthly_savings = self.income - total_spent
        self.current_savings += monthly_savings
        
        # Calculate reward
        reward = self._calculate_reward(actual_spending, monthly_savings)
        
        # Update month
        self.current_month += 1
        done = self.current_month >= self.time_horizon
        
        return self._get_state(), reward, done
    
    def _calculate_reward(self, actual_spending: np.ndarray, monthly_savings: float) -> float:
        """Calculate reward based on spending and savings"""
        # Reward for meeting savings goal
        savings_reward = min(monthly_savings / (self.savings_goal / self.time_horizon), 1.0)
        
        # Penalty for overspending in any category
        overspending_penalty = 0
        for i, cat in enumerate(self.categories):
            historical_avg = self.historical_data[self.historical_data['category'] == cat]['amount'].mean()
            if actual_spending[i] > historical_avg * 1.2:
                overspending_penalty += (actual_spending[i] - historical_avg * 1.2) / self.income
        
        # Reward for balanced spending
        balance_reward = 1 - np.std(actual_spending / self.income)
        
        return savings_reward - overspending_penalty + balance_reward

class BudgetQNetwork(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(BudgetQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class BudgetOptimizer:
    def __init__(self, user_id: str):
        self.user_id = user_id
        logger.info(f"Initializing BudgetOptimizer for user {user_id}")
        
        # Get the models directory from the environment or use a default
        models_dir = os.environ.get('MODELS_DIR')
        if not models_dir:
            logger.warning("MODELS_DIR environment variable not set")
            models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
            os.makedirs(models_dir, exist_ok=True)
            os.environ['MODELS_DIR'] = models_dir
            logger.info(f"Set MODELS_DIR to {models_dir}")
        else:
            logger.info(f"Using MODELS_DIR from environment: {models_dir}")
            
        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
        
        self.models_dir = models_dir
        self.model_path = os.path.join(models_dir, f'budget_optimizer_{user_id}.pt')
        logger.info(f"Model path set to: {self.model_path}")
        
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        # Initialize Q-network
        self.input_size = None  # Will be set during training
        self.output_size = None  # Will be set during training
        self.q_network = None
        self.target_network = None
        self.optimizer = None
        
        # Try to load existing model
        try:
            self._load_model()
        except Exception as e:
            logger.error(f"Error during model initialization: {str(e)}", exc_info=True)
            logger.info("Continuing with uninitialized model")
        
    def _load_model(self):
        """Load existing model if available"""
        try:
            if os.path.exists(self.model_path):
                # Load model configuration
                config_path = self.model_path.replace('.pt', '_config.json')
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        self.input_size = config['input_size']
                        self.output_size = config['output_size']
                        
                    # Initialize and load model
                    self.q_network = BudgetQNetwork(self.input_size, self.output_size)
                    self.q_network.load_state_dict(torch.load(self.model_path))
                    self.q_network.eval()  # Set to evaluation mode
                    logger.info(f"Loaded existing budget optimizer model for user {self.user_id}")
        except Exception as e:
            logger.error(f"Error loading budget optimizer model: {str(e)}")
            self.q_network = None
            
    def train(self, historical_data: pd.DataFrame, categories: List[str], 
              income: float, savings_goal: float, time_horizon: int = 12,
              episodes: int = 1000):
        """Train the budget optimizer"""
        try:
            logger.info(f"Starting budget optimizer training with income={income}, savings_goal={savings_goal}")
            logger.info(f"Data shape: {historical_data.shape}, Categories: {categories}")
            
            if len(historical_data) < 30:
                logger.warning("Not enough data for training (minimum 30 records required)")
                return False
            
            # Check models directory
            models_dir = os.environ.get('MODELS_DIR')
            if not models_dir:
                logger.warning("MODELS_DIR environment variable not set")
                models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
                logger.info(f"Using default models directory: {models_dir}")
            
            os.makedirs(models_dir, exist_ok=True)
            logger.info(f"Ensuring models directory exists: {models_dir}")
            
            try:
                # Verify model path
                self.model_path = os.path.join(models_dir, f'budget_optimizer_{self.user_id}.pt')
                logger.info(f"Model will be saved to: {self.model_path}")
            except Exception as path_err:
                logger.error(f"Error with model path: {str(path_err)}")
                self.model_path = os.path.join(models_dir, f'budget_optimizer_{self.user_id}.pt')
                
            # Initialize environment
            try:
                logger.info("Initializing BudgetEnvironment")
                env = BudgetEnvironment(historical_data, categories, income, savings_goal, time_horizon)
                logger.info("BudgetEnvironment initialized successfully")
            except Exception as env_err:
                logger.error(f"Error initializing environment: {str(env_err)}", exc_info=True)
                raise RuntimeError(f"Failed to initialize training environment: {str(env_err)}")
            
            # Initialize networks
            try:
                logger.info("Initializing neural networks")
                self.input_size = len(categories) + 2
                self.output_size = len(categories)
                logger.info(f"Network dimensions: input_size={self.input_size}, output_size={self.output_size}")
                
                self.q_network = BudgetQNetwork(self.input_size, self.output_size)
                self.target_network = BudgetQNetwork(self.input_size, self.output_size)
                self.target_network.load_state_dict(self.q_network.state_dict())
                self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
                logger.info("Neural networks initialized successfully")
            except Exception as nn_err:
                logger.error(f"Error initializing neural networks: {str(nn_err)}", exc_info=True)
                raise RuntimeError(f"Failed to initialize neural networks: {str(nn_err)}")
            
            # Training loop
            try:
                logger.info(f"Starting training loop with {episodes} episodes")
                for episode in range(episodes):
                    state = env.reset()
                    total_reward = 0
                    done = False
                    
                    while not done:
                        # Select action
                        if random.random() < self.epsilon:
                            action = np.random.dirichlet(np.ones(len(categories)))
                        else:
                            with torch.no_grad():
                                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                                q_values = self.q_network(state_tensor)
                                action = torch.softmax(q_values, dim=1).numpy()[0]
                        
                        # Execute action
                        next_state, reward, done = env.step(action)
                        total_reward += reward
                        
                        # Store experience
                        self.memory.append((state, action, reward, next_state, done))
                        
                        # Train on batch
                        if len(self.memory) >= self.batch_size:
                            self._train_batch()
                        
                        state = next_state
                    
                    # Update target network
                    if episode % 10 == 0:
                        self.target_network.load_state_dict(self.q_network.state_dict())
                    
                    # Decay epsilon
                    self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
                    
                    if (episode + 1) % 100 == 0:
                        logger.info(f"Episode {episode + 1}, Total Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.2f}")
                
                logger.info("Training loop completed successfully")
            except Exception as train_err:
                logger.error(f"Error in training loop: {str(train_err)}", exc_info=True)
                raise RuntimeError(f"Training loop failed: {str(train_err)}")
            
            # Save model and configuration
            try:
                logger.info(f"Saving model to {self.model_path}")
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                torch.save(self.q_network.state_dict(), self.model_path)
                
                # Save configuration
                config = {
                    'input_size': self.input_size,
                    'output_size': self.output_size
                }
                config_path = self.model_path.replace('.pt', '_config.json')
                with open(config_path, 'w') as f:
                    json.dump(config, f)
                logger.info("Model and configuration saved successfully")
            except Exception as save_err:
                logger.error(f"Error saving model: {str(save_err)}", exc_info=True)
                raise RuntimeError(f"Failed to save model: {str(save_err)}")
            
            logger.info("Budget optimizer training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error training budget optimizer: {str(e)}", exc_info=True)
            return False
    
    def _train_batch(self):
        """Train on a batch of experiences"""
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        
        # Current Q values
        current_q_values = self.q_network(states)
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            max_next_q_values = torch.max(next_q_values, dim=1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values
        
        # Compute loss and update
        loss = nn.MSELoss()(current_q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def get_budget_recommendation(self, historical_data: pd.DataFrame, 
                                categories: List[str], income: float) -> Dict[str, float]:
        """Get budget recommendations for the next month"""
        try:
            if historical_data.empty or len(categories) == 0:
                logger.warning("Empty data or categories provided, creating baseline recommendations")
                return self._create_baseline_recommendations(historical_data, categories, income)
                
            if not self.q_network:
                logger.info(f"Checking for model at {self.model_path}")
                if not os.path.exists(self.model_path):
                    # If no model exists, create a simple baseline recommendation
                    logger.info("No trained model available, creating baseline recommendations")
                    return self._create_baseline_recommendations(historical_data, categories, income)
                else:
                    # Try to load the model
                    logger.info("Found model file, attempting to load it")
                    self._load_model()
                    if not self.q_network:
                        logger.warning("Failed to load model, creating baseline recommendations")
                        return self._create_baseline_recommendations(historical_data, categories, income)
            
            # Get current state
            logger.info("Creating environment for recommendations")
            env = BudgetEnvironment(historical_data, categories, income, 0, 1)
            state = env._get_state()
            
            # Get action from Q-network
            logger.info("Getting recommendations from Q-network")
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                action = torch.softmax(q_values, dim=1).numpy()[0]
            
            # Convert to budget recommendations
            recommendations = {}
            for i, cat in enumerate(categories):
                recommendations[cat] = float(action[i] * income)
            
            # Validate recommendations
            if not recommendations or sum(recommendations.values()) == 0:
                logger.warning("Q-network produced empty or zero recommendations, using baseline")
                return self._create_baseline_recommendations(historical_data, categories, income)
                
            logger.info(f"Successfully generated recommendations for {len(categories)} categories")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting budget recommendations: {str(e)}")
            return self._create_baseline_recommendations(historical_data, categories, income)
            
    def _create_baseline_recommendations(self, historical_data: pd.DataFrame,
                                       categories: List[str], income: float) -> Dict[str, float]:
        """Create baseline recommendations based on historical spending patterns"""
        try:
            logger.info("Creating baseline recommendations")
            
            # Handle empty dataframe
            if historical_data.empty:
                logger.warning("Empty dataframe provided for baseline recommendations")
                # Return even distribution
                even_distribution = income / max(len(categories), 1)
                return {cat: float(even_distribution) for cat in categories} if categories else {}
                
            # Handle empty categories list
            if not categories:
                logger.warning("No categories provided for baseline recommendations")
                return {}
            
            # Calculate historical spending patterns
            historical_spending = historical_data.groupby('category')['amount'].sum()
            total_spent = historical_spending.sum()
            
            # If no historical data, distribute evenly
            if total_spent == 0:
                logger.warning("No historical spending found, using even distribution")
                even_distribution = income / len(categories)
                return {cat: float(even_distribution) for cat in categories}
            
            # Calculate proportions
            recommendations = {}
            for cat in categories:
                cat_spent = historical_spending.get(cat, 0)
                proportion = cat_spent / total_spent
                recommendations[cat] = float(proportion * income * 0.9)  # Reserve 10% for savings
            
            logger.info("Successfully created baseline recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error creating baseline recommendations: {str(e)}")
            # Fallback to even distribution
            try:
                even_distribution = income / max(len(categories), 1)
                result = {cat: float(even_distribution) for cat in categories} if categories else {}
                logger.info("Falling back to even distribution after error")
                return result
            except Exception as fallback_error:
                logger.error(f"Critical error in fallback recommendation: {str(fallback_error)}")
                return {} 