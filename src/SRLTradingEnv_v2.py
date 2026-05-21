import gymnasium as gym
import numpy as np
from gymnasium import spaces

class SRLTradingEnv_v2(gym.Env):
    def __init__(self, df_features, df_prices, initial_balance=1000, fee=0.001, asset_type="BTC"):
        super(SRLTradingEnv_v2, self).__init__()
        self.df_features = df_features
        self.df_prices = df_prices
        self.initial_balance = initial_balance
        self.fee = fee 
        self.asset_type = asset_type.upper()
        
        # Asignación dinámica del trade_penalty según el activo real
        # Reducidos a valores lógicos alineados con np.log() para no romper el gradiente
        if self.asset_type == "BTC":
            self.trade_penalty_value = 0.002  # Multa psicológica suave pero firme
        elif self.asset_type == "ETH":
            self.trade_penalty_value = 0.003  # Un poco más alta por el ruido de ETH
        else: # SP500 o SPY
            self.trade_penalty_value = 0.0005
            
        # Parámetro KAPPA (Horizonte mínimo sugerido por Luiz para evitar overtrading)
        self.kappa = 6  # Obliga idealmente a mantener la posición al menos 6 horas
        
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
        self.position = 1 # 0:Short, 1:Out, 2:Long
        self.entry_price = 0
        self.steps_in_position = 0
        return self._get_obs(), {}

    def _get_obs(self):
        embedding = self.df_features.iloc[self.current_step].values
        unrealized_pnl = 0
        current_price = self.df_prices.iloc[self.current_step]
        
        if self.entry_price > 0:
            if self.position == 2: # Long
                unrealized_pnl = (current_price - self.entry_price) / self.entry_price
            elif self.position == 0: # Short
                unrealized_pnl = (self.entry_price - current_price) / self.entry_price
        
        obs = np.append(embedding, [
            float(self.position - 1), 
            float(unrealized_pnl), 
            min(float(self.steps_in_position) / 100.0, 1.0)
        ])
        return obs.astype(np.float32)

    def step(self, action):
        prev_net_worth = self.net_worth
        decision_price = self.df_prices.iloc[self.current_step]
        
        trade_penalty = 0
        
        if action != self.position:
            # 1. Aplicamos la penalización por operar (Alineada dinámicamente)
            trade_penalty = self.trade_penalty_value
            
            # 2. IMPLEMENTACIÓN DE KAPPA: Multa extra por impaciencia (Overtrading)
            if self.steps_in_position < self.kappa and self.position != 1:
                trade_penalty += self.trade_penalty_value * 2.0  # Multa doble si cambia muy rápido
            
            # Liquidar posición anterior al precio actual
            if self.position != 1:
                self.balance = self.net_worth * (1 - self.fee)
            
            # Abrir nueva posición
            self.position = action
            if self.position != 1:
                self.entry_price = decision_price
                self.balance -= self.balance * self.fee
            else:
                self.entry_price = 0
            
            self.steps_in_position = 0
        else:
            if self.position != 1: 
                self.steps_in_position += 1

        # --- AVANCE DEL TIEMPO ---
        self.current_step += 1
        done = self.current_step >= len(self.df_features) - 1
        
        if done:
            return self._get_obs(), 0.0, True, False, {"net_worth": self.net_worth}

        # Actualizar Net Worth con el precio de la nueva vela (t+1)
        next_price = self.df_prices.iloc[self.current_step]
        
        if self.position == 2: # Long
            self.net_worth = self.balance * (next_price / self.entry_price)
        elif self.position == 0: # Short
            pnl_perc = (self.entry_price - next_price) / self.entry_price
            self.net_worth = self.balance * (1 + pnl_perc)
        else:
            self.net_worth = self.balance

        # --- RECOMPENSA LOGARÍTMICA REGULARIZADA ---
        ratio = self.net_worth / prev_net_worth if prev_net_worth > 0 else 1.0
        safe_ratio = max(ratio, 0.0001) 
        
        # Recompensa base
        reward = np.log(safe_ratio)
        
        # Aplicamos la penalización restando directamente del espacio logarítmico calibrado
        reward = reward - trade_penalty
        
        return self._get_obs(), float(reward), done, False, {"net_worth": self.net_worth}