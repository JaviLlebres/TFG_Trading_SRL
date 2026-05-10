import gymnasium as gym
import numpy as np
from gymnasium import spaces

trade_btc = 0.0015
trade_spy = 0.0001

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
        # 1. Guardamos el estado previo para calcular la recompensa después
        prev_net_worth = self.net_worth
        # Precio de decisión (donde el agente está "ahora")
        decision_price = self.df_prices.iloc[self.current_step]
        
        # 2. Lógica de ejecución de órdenes (al precio de cierre de la vela actual)
        trade_penalty = 0
        if action != self.position:
            trade_penalty = trade_spy
            
            # Cerrar anterior
            if self.position != 1:
                self.balance = self.net_worth * (1 - self.fee)
            
            # Abrir nueva
            self.position = action
            if self.position != 1:
                self.entry_price = decision_price
                self.balance -= self.balance * self.fee
            else:
                self.entry_price = 0
            
            self.steps_in_position = 0
        else:
            if self.position != 1: self.steps_in_position += 1

        # --- LA CLAVE: AVANZAMOS EL RELOJ ---
        self.current_step += 1
        
        # Comprobamos si hemos terminado antes de acceder al nuevo precio
        done = self.current_step >= len(self.df_features) - 1
        if done:
            return self._get_obs(), 0, True, False, {"net_worth": self.net_worth}

        # 3. ACTUALIZAR NET WORTH CON EL NUEVO PRECIO (El de la vela siguiente)
        next_price = self.df_prices.iloc[self.current_step]
        
        if self.position == 2: # Long
            self.net_worth = self.balance * (next_price / self.entry_price)
        elif self.position == 0: # Short
            pnl_perc = (self.entry_price - next_price) / self.entry_price
            self.net_worth = self.balance * (1 + pnl_perc)
        else:
            self.net_worth = self.balance

        # 4. RECOMPENSA (Basada en el movimiento que ACABA de ocurrir)
        log_return = np.log(self.net_worth / prev_net_worth) if prev_net_worth > 0 else 0
        reward = (log_return * 5) - trade_penalty
        
        return self._get_obs(), reward, done, False, {"net_worth": self.net_worth}