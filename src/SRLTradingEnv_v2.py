import gymnasium as gym
import numpy as np
from gymnasium import spaces

trade_btc = 0.015
trade_eth = 0.015
trade_spy = 0.002

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
            # El embedding en current_step ya viene shifteado del notebook de preprocesamiento (t-1)
            embedding = self.df_features.iloc[self.current_step].values
            
            unrealized_pnl = 0
            # Usamos el precio en el que estamos (cierre de la vela que acabamos de ver)
            current_price = self.df_prices.iloc[self.current_step]
            
            if self.entry_price > 0:
                if self.position == 2: # Long
                    unrealized_pnl = (current_price - self.entry_price) / self.entry_price
                elif self.position == 0: # Short
                    unrealized_pnl = (self.entry_price - current_price) / self.entry_price
            
            # Escalamiento: steps_in_position / 100.0 es buena idea para mantener valores pequeños
            obs = np.append(embedding, [
                float(self.position - 1), 
                float(unrealized_pnl), 
                min(float(self.steps_in_position) / 100.0, 1.0) # Limitar a 1.0
            ])
            return obs.astype(np.float32)

    def step(self, action):
            prev_net_worth = self.net_worth
            # Precio al que el agente "ve" el mercado para decidir
            decision_price = self.df_prices.iloc[self.current_step]
            
            trade_penalty = 0
            if action != self.position:
                trade_penalty = trade_spy
                
                # 1. Liquidar posición anterior al precio actual (cierre de t)
                if self.position != 1:
                    # El net_worth ya se actualizó en el step anterior con el precio de t
                    self.balance = self.net_worth * (1 - self.fee)
                
                # 2. Abrir nueva posición al precio actual (cierre de t)
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
                return self._get_obs(), 0, True, False, {"net_worth": self.net_worth}

            # 3. ACTUALIZAR NET WORTH CON EL PRECIO DE LA NUEVA VELA (t+1)
            next_price = self.df_prices.iloc[self.current_step]
            
            if self.position == 2: # Long
                self.net_worth = self.balance * (next_price / self.entry_price)
            elif self.position == 0: # Short
                pnl_perc = (self.entry_price - next_price) / self.entry_price
                self.net_worth = self.balance * (1 + pnl_perc)
            else:
                self.net_worth = self.balance

            # --- RECOMPENSA SEGURA ---
            # Calculamos el ratio de cambio de capital
            ratio = self.net_worth / prev_net_worth if prev_net_worth > 0 else 1.0

            # Clip de seguridad: Evitamos que el ratio sea 0 o negativo antes del logaritmo
            # Un ratio de 0.0001 equivale a una pérdida del 99.99% en un solo paso
            safe_ratio = max(ratio, 0.0001) 

            reward = np.log(safe_ratio)
            reward = (reward * 10) - trade_penalty
            
            return self._get_obs(), reward, done, False, {"net_worth": self.net_worth}