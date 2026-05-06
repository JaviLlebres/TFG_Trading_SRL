import gymnasium as gym
import numpy as np
from gymnasium import spaces

class SRLTradingEnv_v2(gym.Env):
    def __init__(self, df_features, df_prices, initial_balance=1000, fee=0.001):
        super(SRLTradingEnv_v2, self).__init__()
        self.df_features = df_features
        self.df_prices = df_prices
        self.initial_balance = initial_balance
        self.fee = fee 
        
        # 3 Acciones (0: Short, 1: Out, 2: Long)
        self.action_space = spaces.Discrete(3)
        
        # Estado: [Embedding + Posición + PnL + Tiempo]
        obs_shape = self.df_features.shape[1] + 3 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.position = 1 # 0:Short, 1:Out, 2:Long para mapear con acciones
        self.entry_price = 0
        self.steps_in_position = 0
        return self._get_obs(), {}

    def _get_obs(self):
        embedding = self.df_features.iloc[self.current_step].values
        unrealized_pnl = 0
        current_price = self.df_prices.iloc[self.current_step]
        
        # Lógica de PnL para Long y Short
        if self.entry_price > 0:
            if self.position == 2: # Long
                unrealized_pnl = (current_price - self.entry_price) / self.entry_price
            elif self.position == 0: # Short
                unrealized_pnl = (self.entry_price - current_price) / self.entry_price
        
        # Estado normalizado
        obs = np.append(embedding, [float(self.position - 1), float(unrealized_pnl), self.steps_in_position / 100.0])
        return obs.astype(np.float32)

    def step(self, action):
        prev_net_worth = self.net_worth
        current_price = self.df_prices.iloc[self.current_step]
        
        # 1. LÓGICA DE TRADING
        trade_penalty = 0
        
        # Si la acción cambia respecto a nuestra posición actual
        if action != self.position:
            # Penalización por operación a cero
            trade_penalty = 0.0
            
            # 1. Cerrar posición anterior si existía (cobrar comisión)
            if self.position != 1: 
                self.balance = self.net_worth * (1 - self.fee)
            
            # 2. Abrir nueva posición
            self.position = action
            if self.position != 1: # Si entramos en Long o Short
                self.entry_price = current_price
                self.balance -= self.balance * self.fee # Comisión de entrada
            else:
                self.entry_price = 0
                
            self.steps_in_position = 0
        else:
            if self.position != 1: self.steps_in_position += 1

        # 2. ACTUALIZAR NET WORTH
        if self.position == 2: # Long
            self.net_worth = self.balance * (current_price / self.entry_price)
        elif self.position == 0: # Short
            # Si el precio baja, el capital sube
            pnl_perc = (self.entry_price - current_price) / self.entry_price
            self.net_worth = self.balance * (1 + pnl_perc)
        else: # Out
            self.net_worth = self.balance

        # 3. RECOMPENSA (Reward Shaping) 
        # Usamos el log-return del Net Worth
        log_return = np.log(self.net_worth / prev_net_worth) if prev_net_worth > 0 else 0
        reward = (log_return * 5) - trade_penalty # Multiplicamos por 5 para dar más señal
        
        # 4. AVANZAR 
        self.current_step += 1
        done = self.current_step >= len(self.df_features) - 1
        
        return self._get_obs(), reward, done, False, {"net_worth": self.net_worth}