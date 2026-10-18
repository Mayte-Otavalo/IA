"""
Modelo de Red Neuronal Multicapa (MLP) y algoritmo de agregacion FedAvg
para el sistema de Aprendizaje Federado sobre Air Quality Dataset.

Arquitectura del MLP (899 parametros entrenables):
  - Entrada: 9 caracteristicas de sensores quimicos y clima
  - Capa Oculta 1: Linear(9 -> 32) + ReLU   (W1: 9x32, b1: 32)
  - Capa Oculta 2: Linear(32 -> 16) + ReLU  (W2: 32x16, b2: 16)
  - Capa Salida:   Linear(16 -> 3) + Softmax (W3: 16x3, b3: 3)
"""

import copy
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

PARAM_KEYS = ["W1", "b1", "W2", "b2", "W3", "b3"]


class FederatedMLP:
    """
    Red Neuronal Multicapa (MLP) implementada explicitamente sobre NumPy
    para garantizar portabilidad total entre las 4 computadoras y transparencia
    absoluta en los pesos intercambiados durante FedAvg.
    """

    def __init__(self, input_dim: int = 9, hidden1: int = 32, hidden2: int = 16, num_classes: int = 3, seed: int = 42):
        self.input_dim = input_dim
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.num_classes = num_classes
        self.seed = seed
        self.params = self._init_weights(seed)

    def _init_weights(self, seed: int) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        # Inicializacion He / Xavier para convergencia estable
        W1 = rng.normal(0.0, np.sqrt(2.0 / self.input_dim), size=(self.input_dim, self.hidden1))
        b1 = np.zeros(self.hidden1, dtype=np.float64)
        W2 = rng.normal(0.0, np.sqrt(2.0 / self.hidden1), size=(self.hidden1, self.hidden2))
        b2 = np.zeros(self.hidden2, dtype=np.float64)
        W3 = rng.normal(0.0, np.sqrt(1.0 / self.hidden2), size=(self.hidden2, self.num_classes))
        b3 = np.zeros(self.num_classes, dtype=np.float64)
        return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}

    def get_weights(self) -> dict[str, list]:
        """Serializa los pesos del modelo a listas nativas para envio por JSON/HTTP."""
        return {k: self.params[k].tolist() for k in PARAM_KEYS}

    def set_weights(self, weights_dict: dict):
        """Carga los pesos recibidos desde el servidor o desde FedAvg."""
        for k in PARAM_KEYS:
            self.params[k] = np.array(weights_dict[k], dtype=np.float64)

    def count_parameters(self) -> int:
        """Retorna el numero total de parametros escalares intercambiados."""
        return int(sum(self.params[k].size for k in PARAM_KEYS))

    @staticmethod
    def _relu(z: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, z)

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, dict]:
        """Paso hacia adelante (Forward Pass) retornando probabilidades y cache de activaciones."""
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]

        z1 = X @ W1 + b1
        a1 = self._relu(z1)
        z2 = a1 @ W2 + b2
        a2 = self._relu(z2)
        logits = a2 @ W3 + b3
        probs = self._softmax(logits)

        cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2, "probs": probs}
        return probs, cache

    def compute_loss(self, probs: np.ndarray, y: np.ndarray, l2_reg: float = 1e-4) -> float:
        """Calcula Cross-Entropy Loss multiclase + regularizacion L2."""
        n = len(y)
        clipped = np.clip(probs[np.arange(n), y], 1e-12, 1.0)
        ce_loss = -np.mean(np.log(clipped))
        l2_term = 0.5 * l2_reg * (
            np.sum(self.params["W1"] ** 2)
            + np.sum(self.params["W2"] ** 2)
            + np.sum(self.params["W3"] ** 2)
        )
        return float(ce_loss + l2_term)

    def train_local(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 5,
        batch_size: int = 64,
        lr: float = 0.03,
        l2_reg: float = 1e-4,
        seed: int = 42,
    ) -> dict:
        """
        Entrena el modelo localmente en los datos privados del cliente (X, y)
        mediante Mini-Batch Stochastic Gradient Descent (SGD).
        """
        rng = np.random.default_rng(seed)
        n = len(X)

        for _ in range(epochs):
            indices = rng.permutation(n)
            X_shuf = X[indices]
            y_shuf = y[indices]

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                xb = X_shuf[start:end]
                yb = y_shuf[start:end]
                m = len(xb)

                probs, cache = self.forward(xb)

                # Gradiente de Cross-Entropy + Softmax respecto a logits
                dz3 = probs.copy()
                dz3[np.arange(m), yb] -= 1.0
                dz3 /= m

                dW3 = cache["a2"].T @ dz3 + l2_reg * self.params["W3"]
                db3 = np.sum(dz3, axis=0)

                da2 = dz3 @ self.params["W3"].T
                dz2 = da2 * (cache["z2"] > 0)
                dW2 = cache["a1"].T @ dz2 + l2_reg * self.params["W2"]
                db2 = np.sum(dz2, axis=0)

                da1 = dz2 @ self.params["W2"].T
                dz1 = da1 * (cache["z1"] > 0)
                dW1 = cache["X"].T @ dz1 + l2_reg * self.params["W1"]
                db1 = np.sum(dz1, axis=0)

                # Actualizacion de parametros locales
                self.params["W1"] -= lr * dW1
                self.params["b1"] -= lr * db1
                self.params["W2"] -= lr * dW2
                self.params["b2"] -= lr * db2
                self.params["W3"] -= lr * dW3
                self.params["b3"] -= lr * db3

        return self.evaluate(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs, _ = self.forward(X)
        return np.argmax(probs, axis=1)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evalua Accuracy, Precision, Recall, F1-score (macro y weighted) y Loss."""
        probs, _ = self.forward(X)
        loss = self.compute_loss(probs, y)
        preds = np.argmax(probs, axis=1)

        acc = float(accuracy_score(y, preds))
        prec_macro = float(precision_score(y, preds, average="macro", zero_division=0))
        rec_macro = float(recall_score(y, preds, average="macro", zero_division=0))
        f1_macro = float(f1_score(y, preds, average="macro", zero_division=0))
        f1_weighted = float(f1_score(y, preds, average="weighted", zero_division=0))
        f1_per_class = f1_score(y, preds, average=None, labels=[0, 1, 2], zero_division=0).tolist()
        cm = confusion_matrix(y, preds, labels=[0, 1, 2]).tolist()

        return {
            "loss": round(loss, 4),
            "accuracy": round(acc, 4),
            "precision": round(prec_macro, 4),
            "recall": round(rec_macro, 4),
            "f1": round(f1_macro, 4),
            "f1_weighted": round(f1_weighted, 4),
            "f1_per_class": [round(v, 4) for v in f1_per_class],
            "confusion_matrix": cm,
        }


def flatten_weights(weights_dict: dict) -> np.ndarray:
    """Convierte un diccionario de pesos en un vector 1D para medir distancia L2 (Client Drift)."""
    return np.concatenate([np.array(weights_dict[k], dtype=np.float64).ravel() for k in PARAM_KEYS])


def fed_avg(client_updates: list[dict]) -> tuple[dict[str, list], dict]:
    """
    Algoritmo Federated Averaging (FedAvg - McMahan et al., 2017).

    Recibe una lista de actualizaciones de los clientes activos en la ronda:
      [
        {"client_id": 1, "num_samples": n_1, "weights": {...}, "metrics": {...}},
        ...
      ]
    Calcula el promedio ponderado por el numero de muestras locales n_k:
      w_global = sum_{k} (n_k / N_total) * w_k

    Retorna:
      - new_global_weights: diccionario de pesos agregados listo para serializar
      - agg_info: metadatos de la agregacion (pesos de ponderacion, deriva de clientes L2)
    """
    if not client_updates:
        raise ValueError("No hay actualizaciones de clientes para agregar en FedAvg.")

    total_samples = sum(int(u["num_samples"]) for u in client_updates)
    aggregated = {
        k: np.zeros_like(np.array(client_updates[0]["weights"][k], dtype=np.float64))
        for k in PARAM_KEYS
    }

    weighting_factors = {}
    for update in client_updates:
        cid = update["client_id"]
        n_k = int(update["num_samples"])
        alpha_k = n_k / total_samples
        weighting_factors[f"client_{cid}"] = round(alpha_k, 4)

        for k in PARAM_KEYS:
            aggregated[k] += alpha_k * np.array(update["weights"][k], dtype=np.float64)

    new_global_weights = {k: aggregated[k].tolist() for k in PARAM_KEYS}

    # Calcular Client Drift (divergencia L2 promedio entre los modelos locales y el modelo global agregado)
    global_vec = flatten_weights(new_global_weights)
    drifts = {}
    for update in client_updates:
        cid = update["client_id"]
        client_vec = flatten_weights(update["weights"])
        drifts[f"client_{cid}"] = round(float(np.linalg.norm(client_vec - global_vec)), 4)

    mean_drift = round(float(np.mean(list(drifts.values()))), 4)
    agg_info = {
        "total_samples": total_samples,
        "participating_clients": [u["client_id"] for u in client_updates],
        "weighting_factors": weighting_factors,
        "client_drift_l2": drifts,
        "mean_client_drift_l2": mean_drift,
    }
    return new_global_weights, agg_info
