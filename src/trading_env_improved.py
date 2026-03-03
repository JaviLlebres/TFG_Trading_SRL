import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from sklearn.preprocessing import StandardScaler

class ImprovedTradingEnv(gym.Env):
    """
    Enhanced trading environment with:
    - Long/Short only (no hold)
    - Proper transaction cost calculation (based on trade size, not net worth)
    - Before/after cost tracking
    - Comprehensive diagnostics
    
    IMPORTANT: Transaction costs are calculated as:
    - Cost per trade = (amount_traded) × (fee_rate)
    - When flipping position: 2 trades (close existing + open new) = 2 × fee
    - Amount traded = net_worth (since we're always fully invested)
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df, initial_balance=1000, transaction_cost=0.0001, lookback_window=20):
        super(ImprovedTradingEnv, self).__init__()
        
        self.df = df
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost  # e.g., 0.0002 = 0.02%
        self.lookback_window = lookback_window
        
        # Initialize scaler for feature normalization
        self.scaler = StandardScaler()
        self.scaler.fit(df.values)
        
        # State variables
        self.net_worth = initial_balance
        self.net_worth_before_costs = initial_balance
        self.prev_net_worth = initial_balance
        self.returns = []
        self.position = 1  # Start with long position (1 = long, -1 = short)
        self.position_duration = 0
        self.total_transaction_costs = 0
        self.trade_count = 0
        
        # Diagnostic tracking
        self.position_history = []
        self.cost_history = []
        self.returns_before_costs = []
        self.returns_after_costs = []
        
        # Start after lookback window
        self.current_step = lookback_window
        self.current_price = self.df.iloc[self.current_step]['close']
        self.entry_price = self.current_price  # Track entry price for position
        
        # Actions: 0 = Long, 1 = Short (no hold - always in the market)
        self.action_space = spaces.Discrete(2)
        
        n_features = len(self.df.columns)
        
        obs_size = (
            1 +  # position (1 or -1)
            1 +  # position_duration
            (lookback_window + 1) * n_features +
            1 +  # net_worth_change
            1 +  # unrealized_pnl
            1 +  # transaction_cost_ratio
            1    # cost_impact (cost as % of recent return)
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        
        print(f"Improved Environment initialized:")
        print(f"  - Features in df: {n_features}")
        print(f"  - Lookback window: {lookback_window}")
        print(f"  - Expected observation size: {obs_size}")
        print(f"  - Action space: 0=Long, 1=Short (ALWAYS IN MARKET)")
        print(f"  - Transaction cost: {transaction_cost*100:.4f}% per trade")
        print(f"  - Cost per position flip: {transaction_cost*2*100:.4f}% (2 trades)")
        print(f"  - Initial position: LONG")
    
    def _get_obs(self):
        """Get current observation with historical context"""
        start_idx = self.current_step - self.lookback_window
        end_idx = self.current_step + 1
        
        historical_data = self.df.iloc[start_idx:end_idx].values
        historical_normalized = self.scaler.transform(historical_data).flatten()
        
        net_worth_change = (self.net_worth - self.prev_net_worth) / self.prev_net_worth if self.prev_net_worth > 0 else 0
        transaction_cost_ratio = self.total_transaction_costs / self.initial_balance
        
        # Unrealized P&L based on current position
        if self.position == 1:
            unrealized_pnl = (self.current_price - self.entry_price) / self.entry_price
        else:  # short
            unrealized_pnl = (self.entry_price - self.current_price) / self.entry_price
        
        # Cost impact: recent cost as % of recent return
        if len(self.cost_history) > 0 and len(self.returns_before_costs) > 0:
            recent_return = abs(self.returns_before_costs[-1]) if self.returns_before_costs[-1] != 0 else 1e-8
            cost_impact = self.cost_history[-1] / recent_return if recent_return != 0 else 0
        else:
            cost_impact = 0
        
        normalized_position = float(self.position)
        normalized_duration = min(self.position_duration / 100.0, 1.0)
        
        obs = np.array([
            normalized_position,
            normalized_duration,
            *historical_normalized,
            net_worth_change,
            unrealized_pnl,
            transaction_cost_ratio,
            cost_impact
        ], dtype=np.float32)
        
        expected_size = self.observation_space.shape[0]
        if obs.shape[0] != expected_size:
            raise ValueError(
                f"Observation size mismatch! Expected {expected_size}, got {obs.shape[0]}"
            )
        
        return obs
    
    def _get_info(self):
        """Get current state information"""
        return {
            "current_date": self.df.index[self.current_step],
            "position": self.position,
            "close_price": self.current_price,
            "entry_price": self.entry_price,
            "net_worth": self.net_worth,
            "net_worth_before_costs": self.net_worth_before_costs,
            "total_transaction_costs": self.total_transaction_costs,
            "position_duration": self.position_duration,
            "trade_count": self.trade_count,
            "cost_per_trade": self.total_transaction_costs / self.trade_count if self.trade_count > 0 else 0
        }
    
    def _calculate_metrics(self):
        """Calculate performance metrics at episode end"""
        # Calculate metrics for both before and after costs
        total_return = (self.net_worth - self.initial_balance) / self.initial_balance
        total_return_before_costs = (self.net_worth_before_costs - self.initial_balance) / self.initial_balance
        
        # Calculate trading frequency
        steps = self.current_step - self.lookback_window
        trade_frequency = self.trade_count / steps if steps > 0 else 0
        
        # Calculate average cost per trade
        avg_cost_per_trade = self.total_transaction_costs / self.trade_count if self.trade_count > 0 else 0
        
        # Cost as percentage of gross returns
        cost_impact_pct = self.total_transaction_costs / abs(self.net_worth_before_costs - self.initial_balance) if self.net_worth_before_costs != self.initial_balance else 0
        
        if len(self.returns) > 1:
            returns_array = np.array(self.returns)
            returns_before_costs_array = np.array(self.returns_before_costs)
            
            # Sharpe ratios
            if np.std(returns_array) > 0:
                sharpe_ratio = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252)
            else:
                sharpe_ratio = 0
                
            if np.std(returns_before_costs_array) > 0:
                sharpe_ratio_before_costs = np.mean(returns_before_costs_array) / np.std(returns_before_costs_array) * np.sqrt(252)
            else:
                sharpe_ratio_before_costs = 0
            
            # Sortino ratios
            negative_returns = returns_array[returns_array < 0]
            if len(negative_returns) > 0 and np.std(negative_returns) > 0:
                sortino_ratio = np.mean(returns_array) / np.std(negative_returns) * np.sqrt(252)
            else:
                sortino_ratio = 0
            
            # Max drawdown
            cumulative_returns = np.cumprod(1 + returns_array)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdown)
            
            # Win rate
            win_rate = np.sum(returns_array > 0) / len(returns_array)
            
            # Calculate win rate before costs
            win_rate_before_costs = np.sum(returns_before_costs_array > 0) / len(returns_before_costs_array)
        else:
            sharpe_ratio = sortino_ratio = max_drawdown = win_rate = 0
            sharpe_ratio_before_costs = win_rate_before_costs = 0
        
        return {
            "total_return": total_return,
            "total_return_before_costs": total_return_before_costs,
            "sharpe_ratio": sharpe_ratio,
            "sharpe_ratio_before_costs": sharpe_ratio_before_costs,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "win_rate_before_costs": win_rate_before_costs,
            "total_transaction_costs": self.total_transaction_costs,
            "trade_count": self.trade_count,
            "trade_frequency": trade_frequency,
            "avg_cost_per_trade": avg_cost_per_trade,
            "cost_impact_pct": cost_impact_pct
        }
    
    def _calculate_reward(self, position_return_before_costs, position_changed, trade_cost_pct):
        """
        Calculate reward with proper accounting
        
        Args:
            position_return_before_costs: Return from price movement
            position_changed: Whether position changed
            trade_cost_pct: Transaction cost as percentage of net worth
        """
        # Start with gross return
        reward = position_return_before_costs
        
        # Subtract transaction costs if trade occurred
        if position_changed:
            reward -= trade_cost_pct
        
        # Optional: Penalize excessive trading
        if position_changed and self.position_duration < 3:
            reward -= 0.0001  # Small penalty for very short positions
        
        return reward
    
    def step(self, action):
        """Execute one step in the environment"""
        self.current_step += 1
        
        if self.current_step >= len(self.df):
            obs = self._get_obs()
            info = self._get_info()
            info.update(self._calculate_metrics())
            return obs, 0, True, False, info
        
        prev_price = self.current_price
        self.current_price = self.df.iloc[self.current_step]['close']
        
        prev_position = self.position
        
        # Map action to position: 0 = Long (1), 1 = Short (-1)
        new_position = 1 if action == 0 else -1
        
        # Calculate price return
        price_change_pct = (self.current_price - prev_price) / prev_price if prev_price > 0 else 0
        
        # CRITICAL: Calculate return based on position we HELD during this period
        # NOT the new position we're taking!
        position_return_before_costs = prev_position * price_change_pct
        
        # Update net worth before costs
        self.net_worth_before_costs *= (1 + position_return_before_costs)
        
        # CRITICAL FIX: net_worth must also be updated with returns!
        # Then costs are deducted from it
        self.net_worth *= (1 + position_return_before_costs)
        
        # Check if position changed
        position_changed = (prev_position != new_position)
        
        # Calculate transaction costs
        trade_cost = 0
        trade_cost_pct = 0
        
        if position_changed:
            # TRANSACTION COST CALCULATION:
            # When we flip position (long -> short or short -> long):
            #   1. Close current position: trade net_worth, pay fee
            #   2. Open new position: trade net_worth, pay fee
            #
            # Each trade costs: amount_traded × fee_rate
            # Total cost = net_worth × fee_rate × 2 (for both trades)
            
            # Calculate cost in dollars
            trade_cost = self.net_worth * self.transaction_cost * 2  # 2 trades per flip
            
            # Calculate cost as percentage (for reward calculation)
            trade_cost_pct = (trade_cost / self.net_worth) if self.net_worth > 0 else 0
            
            # Deduct cost from net worth
            self.net_worth -= trade_cost
            self.total_transaction_costs += trade_cost
            self.trade_count += 1
            
            # Update entry price for new position
            self.entry_price = self.current_price
            
            # Reset position duration
            self.position_duration = 0
        else:
            self.position_duration += 1
        
        # Update position
        self.position = new_position
        
        # Calculate actual return after costs
        # The net_worth already has costs deducted, so calculate the actual return
        current_return = (self.net_worth - self.prev_net_worth) / self.prev_net_worth if self.prev_net_worth > 0 else 0
        
        # Track returns
        self.returns_before_costs.append(position_return_before_costs)
        self.returns_after_costs.append(current_return)  # This includes cost impact
        self.returns.append(current_return)
        self.prev_net_worth = self.net_worth
        
        # Track diagnostics
        self.position_history.append(self.position)
        self.cost_history.append(trade_cost_pct)
        
        # Calculate reward
        reward = self._calculate_reward(position_return_before_costs, position_changed, trade_cost_pct)
        
        done = self.current_step >= len(self.df) - 1
        
        obs = self._get_obs()
        info = self._get_info()
        
        if done:
            info.update(self._calculate_metrics())
        
        return obs, reward, done, False, info

    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.net_worth = self.initial_balance
        self.net_worth_before_costs = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.returns = []
        self.returns_before_costs = []
        self.returns_after_costs = []
        self.position = 1  # Start with long position
        self.position_duration = 0
        self.current_step = self.lookback_window
        self.current_price = self.df.iloc[self.current_step]['close']
        self.entry_price = self.current_price
        self.total_transaction_costs = 0
        self.trade_count = 0
        
        # Reset diagnostics
        self.position_history = []
        self.cost_history = []
        
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def render(self, mode='human'):
        """Render current state"""
        position_str = {1: "Long", -1: "Short"}
        print(f'Step: {self.current_step}, '
              f'Net Worth: {self.net_worth:.2f} (before costs: {self.net_worth_before_costs:.2f}), '
              f'Position: {position_str[self.position]} (Duration: {self.position_duration}), '
              f'Price: {self.current_price:.2f}, '
              f'Trades: {self.trade_count}, '
              f'Total Costs: {self.total_transaction_costs:.2f}')
    
    def get_diagnostics(self):
        """Get comprehensive diagnostics for analysis"""
        return {
            'position_history': np.array(self.position_history),
            'cost_history': np.array(self.cost_history),
            'returns_before_costs': np.array(self.returns_before_costs),
            'returns_after_costs': np.array(self.returns_after_costs),
            'trade_count': self.trade_count,
            'total_costs': self.total_transaction_costs,
            'net_worth_final': self.net_worth,
            'net_worth_before_costs_final': self.net_worth_before_costs
        }