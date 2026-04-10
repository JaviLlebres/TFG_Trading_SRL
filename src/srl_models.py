import torch
import torch.nn as nn

class TemporalAutoencoder(nn.Module):
    """
    Autoencoder basado en LSTMs para State Representation Learning (SRL).
    Diseñado para comprimir secuencias temporales en un embedding latente.
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=1):

        """
        Inicializa la arquitectura del Autoencoder Temporal.
    
        Configura las capas LSTM para el procesamiento de secuencias y las capas lineales 
        la proyección al espacio latente. Define la capacidad de la red 
        para comprimir información multidimensional en un vector de características denso.
        """

        super(TemporalAutoencoder, self).__init__()
        
        # Comprime la secuencia en un vector (Embedding)
        self.encoder = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True
        )
        
        # Capa lineal para llegar al espacio latente (el tamaño del embedding final)
        self.latent_layer = nn.Linear(hidden_dim, latent_dim)
        
        # Reconstruye la secuencia original a partir del embedding
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True
        )
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):

        """
        Extrae la representación latente (embedding) de una secuencia temporal de entrada.
        
        Procesa la ventana de datos a través de una red LSTM y utiliza el último estado 
        oculto para generar un resumen matemático del estado del mercado.
        """

        # x shape: (batch, seq_len, input_dim)
        _, (hidden, _) = self.encoder(x)
        # Tomamos el último estado oculto para crear el embedding
        latent = self.latent_layer(hidden[-1]) 
        return latent

    def decode(self, latent, seq_len):

        """
        Reconstruye la secuencia temporal original a partir del vector latente.
        
        Obliga al encoder a retener solo la información más relevante necesaria 
        para recrear la señal de entrada sin ruido aleatorio.
        """ 

        # Repetimos el vector latente para cada paso de tiempo de la secuencia
        x = self.decoder_input(latent).unsqueeze(1).repeat(1, seq_len, 1)
        x, _ = self.decoder_lstm(x)
        return self.output_layer(x)

    def forward(self, x):

        """
        Ejecuta el flujo completo de propagación hacia adelante del Autoencoder.
        
        Integra las fases de codificación y decodificación para permitir el cálculo del 
        error de reconstrucción durante el entrenamiento y la extracción simultánea de 
        los embeddings latentes.
        """

        seq_len = x.size(1)
        latent = self.encode(x)
        reconstruction = self.decode(latent, seq_len)
        return reconstruction, latent
    

class CPCModel(nn.Module):
    """
    Implementación corregida de Contrastive Predictive Coding (CPC).
    """
    def __init__(self, input_dim, enc_dim, context_dim, predict_steps):
        super(CPCModel, self).__init__()
        self.enc_dim = enc_dim
        self.context_dim = context_dim
        self.predict_steps = predict_steps

        # 1. Encoder (g_enc): Comprime la entrada actual
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, enc_dim)
        )

        # 2. Modelo Autoregresivo (g_ar): Resume la historia en un contexto c_t
        self.gru = nn.GRU(input_size=enc_dim, hidden_size=context_dim, batch_first=True)

        # 3. Predictores (W_k): Intentan predecir z_{t+k} a partir de c_t
        self.predictors = nn.ModuleList([
            nn.Linear(context_dim, enc_dim) for _ in range(predict_steps)
        ])

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # 1. Pasar cada paso de tiempo por el encoder
        x_flat = x.reshape(-1, x.shape[-1]) 
        z = self.encoder(x_flat)
        z = z.reshape(batch_size, seq_len, self.enc_dim) 

        # 2. Pasar la secuencia de z_t por la GRU para obtener c_t
        out_gru, _ = self.gru(z) 
        
        # El contexto actual es el último paso de la secuencia procesada
        c_t = out_gru[:, -1, :] 

        return z, c_t

    def predict_latents(self, c_t):
        predictions = []
        for i in range(self.predict_steps):
            predictions.append(self.predictors[i](c_t))
        return predictions