import gymnasium as gym
import numpy as np
from gymnasium import spaces

class SRLTradingEnv(gym.Env):
    def __init__(self, df_features, df_prices, initial_balance=1000, fee=0.001):
        super(SRLTradingEnv, self).__init__()
        self.df_features = df_features
        self.df_prices = df_prices
        self.initial_balance = initial_balance
        self.fee = fee 
        
        self.action_space = spaces.Discrete(2)
        # Añadimos un dato extra: PnL acumulado de la posición actual
        obs_shape = self.df_features.shape[1] + 3 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.position = 0 
        self.entry_price = 0
        self.steps_in_position = 0
        return self._get_obs(), {}

    def _get_obs(self):
        # Tomamos el embedding y calculamos el PnL actual
        embedding = self.df_features.iloc[self.current_step].values
        unrealized_pnl = 0
        if self.position == 1:
            current_price = self.df_prices.iloc[self.current_step]
            unrealized_pnl = (current_price - self.entry_price) / self.entry_price
        
        # Estado: [Embedding, Posición, PnL Unrealized, Tiempo]
        obs = np.append(embedding, [float(self.position), float(unrealized_pnl), self.steps_in_position / 100.0])
        return obs.astype(np.float32)

    def step(self, action):
        prev_net_worth = self.net_worth
        current_price = self.df_prices.iloc[self.current_step]
        
        # 1. LÓGICA DE TRADING 
        trade_penalty = 0
        if action != self.position:
            # PENALIZACIÓN DISUASORIA: 0.5% (Evita el overtrading)
            trade_penalty = 0.002
            
            if action == 1: # COMPRAR
                self.position = 1
                self.entry_price = current_price
                self.balance -= self.balance * self.fee
            else: # VENDER
                self.balance = self.balance * (current_price / self.entry_price)
                self.balance -= self.balance * self.fee
                self.position = 0
                self.entry_price = 0
            self.steps_in_position = 0
        else:
            if self.position == 1: self.steps_in_position += 1

        # 2. ACTUALIZAR NET WORTH 
        if self.position == 1:
            self.net_worth = self.balance * (current_price / self.entry_price)
        else:
            self.net_worth = self.balance

        # 3. RECOMPENSA 
        # Solo premiamos el crecimiento del capital y castigamos el exceso de clics
        log_return = np.log(self.net_worth / prev_net_worth) if prev_net_worth > 0 else 0
        
        # Eliminamos el drawdown para evitar ventas en pánico
        reward = (log_return*2) - trade_penalty
        
        # 4. AVANZAR 
        self.current_step += 1
        done = self.current_step >= len(self.df_features) - 1
        
        return self._get_obs(), reward, done, False, {"net_worth": self.net_worth}