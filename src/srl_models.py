import torch
import torch.nn as nn
import math

class TemporalAutoencoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, latent_dim=64, num_layers=2):
        super(TemporalAutoencoder, self).__init__()
        
        # Encoder: Comprime 3 variables x 168 horas
        # CORRECCIÓN: Quitamos el ", _" del final de la asignación
        self.encoder = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        self.latent_layer = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder: Reconstruye la secuencia original
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        
        # CORRECCIÓN: Quitamos el ", _" aquí también
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        # x shape: (batch, seq_len, input_dim)
        _, (hidden, _) = self.encoder(x)
        # Tomamos el último estado oculto de la última capa (hidden[-1])
        latent = self.latent_layer(hidden[-1]) 
        return latent

    def decode(self, latent, seq_len):
        # Repetimos el vector latente para cada paso de tiempo
        x = self.decoder_input(latent).unsqueeze(1).repeat(1, seq_len, 1)
        x, _ = self.decoder_lstm(x)
        return self.output_layer(x)

    def forward(self, x):
        seq_len = x.size(1)
        latent = self.encode(x)
        reconstruction = self.decode(latent, seq_len)
        return reconstruction, latent


class CPCModel(nn.Module):
    def __init__(self, input_dim=3, enc_dim=128, context_dim=256, predict_steps=8):
        super(CPCModel, self).__init__()
        self.enc_dim = enc_dim
        self.context_dim = context_dim
        self.predict_steps = predict_steps

        # 1. Encoder Convolucional (g_enc): Extrae patrones locales
        # Recibe 3 variables (Precio, Rango, Trades)
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, enc_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(enc_dim),
            nn.ReLU()
        )

        # 2. Modelo Autoregresivo (g_ar): GRU para el contexto global
        self.gru = nn.GRU(input_size=enc_dim, hidden_size=context_dim, batch_first=True)

        # 3. Predictores (W_k): Intentan predecir z_{t+k}
        self.predictors = nn.ModuleList([
            nn.Linear(context_dim, enc_dim) for _ in range(predict_steps)
        ])

    def forward(self, x):
        # x shape: [batch, seq_len, input_dim]
        batch_size, seq_len, _ = x.shape
        
        # Ajustar para Conv1d: [batch, input_dim, seq_len]
        x_conv = x.transpose(1, 2)
        z = self.encoder(x_conv)
        
        # Volver a [batch, seq_len, enc_dim] para la GRU
        z = z.transpose(1, 2)

        # Obtener contexto c_t
        out_gru, _ = self.gru(z)
        c_t = out_gru[:, -1, :] 

        return z, c_t

    def predict_latents(self, c_t):
        return [pred(c_t) for pred in self.predictors]


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
    def __init__(self, input_dim=3, embed_dim=128, nhead=8, num_layers=4, dropout=0.1):
        super(MaskedTransformerSRL, self).__init__()
        self.embed_dim = embed_dim

        # Proyección de las 3 variables
        self.input_proj = nn.Linear(input_dim, embed_dim)
        
        # Token [CLS] para resumir la secuencia
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional Encoding (asegúrate de que sea la clase que ya tenías)
        self.pos_encoder = PositionalEncoding(embed_dim)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=nhead, 
            dim_feedforward=embed_dim * 4, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # Decoder para reconstruir las 3 variables
        self.decoder = nn.Linear(embed_dim, input_dim)

    def forward(self, x):
        batch_size = x.size(0)
        
        # 1. Proyectar entrada
        x = self.input_proj(x) * math.sqrt(self.embed_dim)
        
        # 2. Añadir Token [CLS] al inicio
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) 
        
        # 3. Posiciones y Transformer
        x = self.pos_encoder(x)
        latent_sequence = self.transformer_encoder(x)
        
        # 4. Extraer el primer token (el resumen CLS) para el Agente RL
        latent_vector = latent_sequence[:, 0, :]
        
        # 5. Reconstruir la secuencia original (sin el CLS)
        reconstructed = self.decoder(latent_sequence[:, 1:, :])
        
        return reconstructed, latent_vector