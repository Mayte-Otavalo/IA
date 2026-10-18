
# Examen Práctico: Aprendizaje Federado IID vs. non-IID (Air Quality Dataset)

Sistema distribuido de **Aprendizaje Federado (`FedAvg`)** implementado sobre una arquitectura Cliente-Servidor HTTP/REST en red local para 4 computadoras, utilizando el dataset **[Air Quality Data Set (UCI / Kaggle)](https://www.kaggle.com/datasets/fedesoriano/air-quality-data-set/data)**.

---

## 1. Arquitectura Distribuida (4 Computadoras en Red Local)

```text
                        COMPUTADORA 1: SERVIDOR
                      server.py (Puerto TCP 5000)
                      Modelo Global MLP (899 params)
                        Dashboard Web en Vivo
                                  |
             +--------------------+--------------------+
             |                    |                    |
      COMPUTADORA 2        COMPUTADORA 3        COMPUTADORA 4
        Cliente 1            Cliente 2            Cliente 3
     (client_1.csv)       (client_2.csv)       (client_3.csv)
      Modelo Local         Modelo Local         Modelo Local
             \                    |                    /
              +---------> Agregación FedAvg <---------+
                                  |
                        Nuevo Modelo Global
```

- **Privacidad de los datos**: Los datos crudos (`data/iid/client_k.csv` y `data/non_iid/client_k.csv`) permanecen exclusivamente en cada computadora cliente. **Jamás se transmiten filas ni características al servidor.**
- **Parámetros intercambiados**:
  - **Servidor $\rightarrow$ Clientes**: Diccionario JSON con las matrices de pesos y vectores de sesgo de las 3 capas del perceptrón multicapa (`W1` $[9\times 32]$, `b1` $[32]$, `W2` $[32\times 16]$, `b2` $[16]$, `W3` $[16\times 3]$, `b3` $[3]$, total = **899 parámetros escalares**).
  - **Clientes $\rightarrow$ Servidor**: Pesos entrenados localmente $\mathbf{w}_k^{(t)}$, tamaño de su dataset local $n_k$ y métricas locales (`accuracy`, `precision`, `recall`, `f1`, `loss`).
- **Estrategia de agregación (`FedAvg`)**:
  $$\mathbf{w}^{(t+1)} = \sum_{k \in S_t} \frac{n_k}{\sum_{j \in S_t} n_j} \mathbf{w}_k^{(t+1)}$$
  donde $S_t$ es el subconjunto de clientes conectados que respondieron dentro del tiempo límite (`round_timeout`) en la ronda $t$.

---

## 2. Requisitos e Instalación Rápida (en las 4 Computadoras)

Copien esta carpeta (`tarea_grupal`) en las 4 computadoras. Solo se requieren librerías estándar de ciencia de datos en Python:

```powershell
py -m pip install numpy pandas scikit-learn matplotlib requests
```

Si aún no se han generado las particiones de datos, ejecuten una sola vez:

```powershell
py prepare_data.py
```

---

## 3. Instrucciones Paso a Paso para Ejecutar entre las 4 Computadoras

### Paso 1: En la Computadora 1 (Servidor Federado)
1. Obtener la dirección IP local (Wi-Fi / LAN) de la Computadora 1:
   ```powershell
   ipconfig
   ```
   *(Ejemplo: `192.168.1.105` — el servidor también la imprime automáticamente al arrancar).*
2. *(Opcional si el Firewall de Windows bloquea conexiones entrantes)* Abrir el puerto `5000` en PowerShell como Administrador:
   ```powershell
   New-NetFirewallRule -DisplayName "Servidor Federado 5000" -Direction Inbound -Protocol TCP -LocalPort 5000 -Action Allow
   ```
3. Iniciar el servidor para el **Experimento 1 (IID)**:
   ```powershell
   py server.py --host 0.0.0.0 --port 5000 --rounds 15 --round-timeout 12 --scenario iid
   ```
4. Abrir en el navegador el **Dashboard en Vivo** para proyectar a la clase:
   - `http://localhost:5000` (o `http://<IP_SERVIDOR>:5000`)

---

### Paso 2: En las Computadoras 2, 3 y 4 (Clientes 1, 2 y 3)
Reemplacen `<IP_SERVIDOR>` por la IP real de la Computadora 1 (ej. `http://192.168.1.105:5000`):

- **Computadora 2 (Estudiante 2 — Cliente 1)**:
  ```powershell
  py client.py --client-id 1 --server http://<IP_SERVIDOR>:5000 --mode iid
  ```
- **Computadora 3 (Estudiante 3 — Cliente 2)**:
  ```powershell
  py client.py --client-id 2 --server http://<IP_SERVIDOR>:5000 --mode iid
  ```
- **Computadora 4 (Estudiante 4 — Cliente 3)**:
  ```powershell
  py client.py --client-id 3 --server http://<IP_SERVIDOR>:5000 --mode iid
  ```

> En cuanto el tercer cliente se registra, el servidor inicia automáticamente las 15 rondas federadas (o pueden pulsar el botón **"Iniciar / Forzar Ronda"** en el Dashboard Web).

---

### Paso 3: Ejecutar el Experimento 2 (Escenario non-IID)
1. En la **Computadora 1 (Servidor)**:
   ```powershell
   py server.py --host 0.0.0.0 --port 5000 --rounds 15 --round-timeout 12 --scenario non_iid
   ```
2. En las **Computadoras 2, 3 y 4 (Clientes)** cambiar `--mode iid` por `--mode non_iid`:
   ```powershell
   # Computadora 2:
   py client.py --client-id 1 --server http://<IP_SERVIDOR>:5000 --mode non_iid

   # Computadora 3:
   py client.py --client-id 2 --server http://<IP_SERVIDOR>:5000 --mode non_iid

   # Computadora 4:
   py client.py --client-id 3 --server http://<IP_SERVIDOR>:5000 --mode non_iid
   ```

---

## 4. Prueba de Desconexión en Vivo (Evaluación del Profesor)

Cuando el profesor desactive temporalmente el Wi-Fi o la tarjeta de red de cualquiera de las computadoras clientes durante la ejecución:

1. **¿Qué cliente dejó de responder?**
   - El hilo supervisor del servidor detecta la ausencia de pulsos (`/heartbeat`) en `5s` y, al cumplirse el `--round-timeout` (`12s`), muestra en consola y en rojo en el Dashboard Web:
     `[ALERTA DESCONEXION] Ronda t finalizada por TIMEOUT: Clientes sin respuesta -> [k].`
2. **¿Qué ocurre con la ronda y si el servidor continúa funcionando?**
   - El servidor **NO se queda esperando indefinidamente**. Cierra la ronda al cumplirse el timeout y recalcula `FedAvg` re-normalizando los pesos $\alpha_j = \frac{n_j}{\sum_{i \in S_t} n_i}$ únicamente entre los clientes disponibles (ej. los otros 2 clientes conectados).
3. **¿Qué ocurre cuando el cliente vuelve a conectarse?**
   - El proceso `client.py` nunca se cierra durante el corte de red (captura la excepción de red y reintenta cada `1.5s`).
   - En cuanto el profesor reactiva el Wi-Fi/red del cliente, el servidor registra:
     `[RECONEXION / AGREGACION] ¡Cliente k ha vuelto a conectarse en la Ronda t! Enviando pesos globales sincronizados...`
   - El cliente descarga los pesos globales más recientes $\mathbf{w}^{(t)}$ de la ronda en curso, entrena localmente y se reincorpora inmediatamente a la agregación `FedAvg`.

---

## 5. Ejecución Automatizada (Todo en una sola computadora para pruebas o generación de gráficas)

Para reproducir los 3 experimentos completos (**IID**, **non-IID** y **Prueba de Desconexión**) en una sola máquina lanzando los procesos servidor y 3 clientes automáticamente:

```powershell
py run_experiments.py
```

Esto genera en `results/`:
- `results/distribucion_iid_vs_non_iid.png` y `results/tabla_distribucion_clientes.csv`
- `results/evolucion_iid_vs_non_iid.png`
- `results/comportamiento_clientes_locales.png`
- `results/prueba_desconexion_recuperacion.png`
- `results/tabla_comparativa_rondas.csv` y `results/tabla_resumen_final.csv`
- `Experimento_Federado_IID_vs_NonIID.ipynb` (Notebook completo con análisis y conclusiones)
