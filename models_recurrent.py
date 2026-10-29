"""
Arquitecturas Recurrentes (LSTM y GRU) para Predicción de Calidad del Aire (CO(GT))
Diseñadas para entrenamiento centralizado y Aprendizaje Federado (FedAvg / FedMD).
"""

import torch
import torch.nn as nn

class LSTMRegressor(nn.Module):
    """
    Red Neuronal Recurrente LSTM para Regresión Temporal de Monóxido de Carbono.
    Entrada: Tensor de dimensión (batch_size, seq_len=24, input_dim=18)
    Salida: Tensor de dimensión (batch_size,) con la concentración predicha de CO(GT).
    """
    def __init__(
        self,
        input_dim: int = 18,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * num_directions, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # Tomar la representación del último paso temporal
        last_step = lstm_out[:, -1, :]
        out = self.head(last_step)
        return out.squeeze(-1)


class GRURegressor(nn.Module):
    """
    Red Neuronal Recurrente GRU (Gated Recurrent Unit) para Regresión Temporal.
    Estructura más liviana (~25% menos parámetros que LSTM), óptima para nodos edge.
    Entrada: (batch_size, seq_len=24, input_dim=18)
    Salida: (batch_size,) con la predicción de CO(GT).
    """
    def __init__(
        self,
        input_dim: int = 18,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * num_directions, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_out, _ = self.gru(x)
        last_step = gru_out[:, -1, :]
        out = self.head(last_step)
        return out.squeeze(-1)


def build_recurrent_model(arch: str = "lstm", input_dim: int = 18) -> nn.Module:
    arch = arch.lower()
    if arch == "lstm":
        return LSTMRegressor(input_dim=input_dim, hidden_dim=64, num_layers=2)
    elif arch == "gru":
        return GRURegressor(input_dim=input_dim, hidden_dim=64, num_layers=2)
    elif arch == "bilstm":
        return LSTMRegressor(input_dim=input_dim, hidden_dim=48, num_layers=2, bidirectional=True)
    elif arch == "bigru":
        return GRURegressor(input_dim=input_dim, hidden_dim=48, num_layers=2, bidirectional=True)
    else:
        raise ValueError(f"Arquitectura recurrente desconocida: {arch}. Opciones: lstm, gru, bilstm, bigru")


if __name__ == "__main__":
    for name in ["lstm", "gru", "bilstm", "bigru"]:
        m = build_recurrent_model(name)
        dummy_in = torch.randn(16, 24, 18)
        pred = m(dummy_in)
        n_params = sum(p.numel() for p in m.parameters())
        print(f"Modelo {name.upper():<7} | Params: {n_params:>6,} | Out shape: {list(pred.shape)}")
