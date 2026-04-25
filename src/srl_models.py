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
    

    import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Inyecta información sobre la posición relativa o absoluta de los tokens en la secuencia.
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return x

class MaskedTransformerSRL(nn.Module):
    """
    Transformer Encoder para State Representation Learning.
    Aprende representaciones mediante la reconstrucción de partes enmascaradas de la serie.
    """
    def __init__(self, input_dim, embed_dim, nhead, num_layers, dropout=0.1):
        super(MaskedTransformerSRL, self).__init__()
        
        self.model_type = 'Transformer'
        self.embed_dim = embed_dim

        # 1. Proyección lineal de entrada (de 19 dimensiones a embed_dim)
        self.input_proj = nn.Linear(input_dim, embed_dim)
        
        # 2. Codificación Posicional
        self.pos_encoder = PositionalEncoding(embed_dim)

        # 3. Encoder de Transformer
        # Usamos batch_first=True para mantener la consistencia con tus otros modelos
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=nhead, 
            dim_feedforward=embed_dim * 4, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # 4. Cabezal de reconstrucción (Decoder)
        self.decoder = nn.Linear(embed_dim, input_dim)

    def forward(self, x):
        """
        Args:
            x: Tensor de entrada [batch_size, seq_len, input_dim]
        Returns:
            reconstructed: La señal reconstruida para calcular el MSE.
            latent: El vector de estado (embedding) extraído.
        """
        # Proyectar características a la dimensión del modelo
        x = self.input_proj(x) * math.sqrt(self.embed_dim)
        
        # Añadir información posicional
        x = self.pos_encoder(x)
        
        # Pasar por las capas de Atención
        # latent_sequence shape: [batch_size, seq_len, embed_dim]
        latent_sequence = self.transformer_encoder(x)
        
        # Para el embedding final (SRL), tomamos la media de la secuencia
        # Esto resume la "atención global" de la ventana de 24h en un solo vector
        latent_vector = latent_sequence.mean(dim=1)
        
        # Reconstrucción (solo para el entrenamiento enmascarado)
        reconstructed = self.decoder(latent_sequence)
        
        return reconstructed, latent_vector