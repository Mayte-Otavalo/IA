"""
Servicio de Archivos e I/O para la Raspberry Pi (FILE / I/O SERVICE).
Implementa la arquitectura de las Diapositivas 3, 4, 5, 6 y 9:
- Datasets locales en CSV y Parquet.
- Patrones de acceso: lectura completa (full), columnas seleccionadas (columns), rango de filas (range) y escritura (write).
- Endpoints equivalentes en Sockets TCP (reutilizando protocol.py) y gRPC.
- Monitoreo a nivel de Sistema Operativo (CPU, RAM, tiempo puro de Disco I/O y bytes de disco).
- Simulación y registro de fallos/reinicio del servicio para el Escenario E.
"""

import os
import sys
import time
import json
import socket
import logging
import argparse
import threading
from concurrent import futures
from typing import Dict, Any, Optional

import pandas as pd

# Asegurar que el directorio raíz esté en sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from protocol import send_frame, recv_frame
from server.fault_tolerance import FaultToleranceManager
from prepare_datasets import prepare_all_datasets, STORAGE_DIR

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import grpc
    HAS_GRPC = True
except ImportError:
    HAS_GRPC = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("PiFileService")


class OSResourceMonitor:
    """Mide recursos del Sistema Operativo en la Raspberry Pi (CPU, RAM, Disco I/O)."""

    def __init__(self):
        self.process = psutil.Process(os.getpid()) if HAS_PSUTIL else None
        if self.process:
            self.process.cpu_percent(interval=None)

    def snapshot_start(self) -> Dict[str, Any]:
        snap = {"t0": time.perf_counter(), "disk_read_0": 0, "disk_write_0": 0}
        if HAS_PSUTIL:
            try:
                io = psutil.disk_io_counters()
                if io:
                    snap["disk_read_0"] = io.read_bytes
                    snap["disk_write_0"] = io.write_bytes
            except Exception:
                pass
        return snap

    def snapshot_end(self, start_snap: Dict[str, Any]) -> Dict[str, float]:
        cpu_pct = 0.0
        ram_mb = 0.0
        disk_read_delta = 0
        disk_write_delta = 0

        if HAS_PSUTIL and self.process:
            try:
                cpu_pct = psutil.cpu_percent(interval=None)
                ram_mb = self.process.memory_info().rss / (1024.0 * 1024.0)
                io = psutil.disk_io_counters()
                if io:
                    disk_read_delta = max(0, io.read_bytes - start_snap.get("disk_read_0", 0))
                    disk_write_delta = max(0, io.write_bytes - start_snap.get("disk_write_0", 0))
            except Exception:
                pass

        return {
            "server_cpu_percent": round(cpu_pct, 2),
            "server_ram_mb": round(ram_mb, 2),
            "os_disk_read_bytes": int(disk_read_delta),
            "os_disk_write_bytes": int(disk_write_delta)
        }


class DistributedFileService:
    """Lógica central del Servicio de Archivos e I/O en la Raspberry Pi."""

    def __init__(self, storage_dir: str = STORAGE_DIR):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.monitor = OSResourceMonitor()
        self.fault_tolerance = FaultToleranceManager()
        self.lock = threading.Lock()
        self.simulated_outage_until = 0.0

        # Verificar que existan los datasets CSV y Parquet; si no, generarlos
        small_csv = os.path.join(self.storage_dir, "dataset_small.csv")
        small_pq = os.path.join(self.storage_dir, "dataset_small.parquet")
        if not (os.path.exists(small_csv) and os.path.exists(small_pq)):
            logger.info("Generando datasets base CSV y Parquet en la Raspberry Pi...")
            prepare_all_datasets(self.storage_dir)

    def _get_dataset_path(self, size_label: str, file_format: str) -> str:
        ext = "parquet" if file_format.lower() == "parquet" else "csv"
        return os.path.join(self.storage_dir, f"dataset_{size_label}.{ext}")

    def execute_operation(self, req: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecuta una operación de almacenamiento/I/O solicitada por el contenedor/PC
        y devuelve los datos junto con las métricas de Disco I/O y Sistema Operativo.
        """
        op = req.get("operation", "ping")

        # Verificar si hay un fallo simulado activo (Escenario E)
        now = time.time()
        if now < self.simulated_outage_until and op != "recover_service":
            remaining = self.simulated_outage_until - now
            self.fault_tolerance.log_failure(
                "SERVICE_UNAVAILABLE", "RaspberryPi_FileService",
                f"Petición '{op}' rechazada durante caída simulada ({remaining:.1f}s restantes)"
            )
            raise ConnectionError("Servicio de archivos en Raspberry Pi temporalmente detenido (Simulated Failure)")

        if op == "ping":
            return {"status": "success", "message": "pong", "timestamp": now}

        if op == "simulate_failure":
            duration = float(req.get("duration_sec", 5.0))
            self.simulated_outage_until = time.time() + duration
            self.fault_tolerance.log_failure(
                "SIMULATED_STOP", "RaspberryPi_FileService",
                f"Servicio detenido por {duration}s para prueba de tolerancia a fallos", duration
            )
            logger.warning(f"[Escenario E] Servicio de archivos en pausa por {duration}s...")
            return {"status": "success", "message": f"Outage simulated for {duration}s"}

        if op == "recover_service":
            self.simulated_outage_until = 0.0
            self.fault_tolerance.log_event("SERVICE_RECOVERED", {"timestamp": time.time()})
            return {"status": "success", "message": "Service recovered immediately"}

        size_label = req.get("size_label", "small")
        file_format = req.get("file_format", "csv").lower()
        filepath = self._get_dataset_path(size_label, file_format)

        os_snap_start = self.monitor.snapshot_start()
        t_io_start = time.perf_counter()

        try:
            if op == "read_full":
                if file_format == "csv":
                    df = pd.read_csv(filepath)
                else:
                    df = pd.read_parquet(filepath)

            elif op == "read_columns":
                columns = req.get("columns", ["CO(GT)", "T", "RH"])
                if file_format == "csv":
                    df = pd.read_csv(filepath, usecols=columns)
                else:
                    df = pd.read_parquet(filepath, columns=columns)

            elif op == "read_range":
                start_row = int(req.get("start_row", 0))
                end_row = int(req.get("end_row", 200))
                nrows = max(1, end_row - start_row)
                if file_format == "csv":
                    df = pd.read_csv(filepath, skiprows=range(1, start_row + 1), nrows=nrows)
                else:
                    df_full = pd.read_parquet(filepath)
                    df = df_full.iloc[start_row:end_row]

            elif op == "write_file":
                records = req.get("records")
                if records:
                    df = pd.DataFrame(records)
                else:
                    # Si no se envían registros, usar el dataset en memoria equivalente para medir I/O puro de escritura
                    src_path = self._get_dataset_path(size_label, "parquet")
                    df = pd.read_parquet(src_path)
                    t_io_start = time.perf_counter()

                write_path = os.path.join(
                    self.storage_dir,
                    f"write_test_{size_label}_{threading.get_ident()}.{file_format}"
                )
                with self.lock:
                    if file_format == "csv":
                        df.to_csv(write_path, index=False)
                    else:
                        df.to_parquet(write_path, index=False, compression="snappy")
                    file_size_bytes = os.path.getsize(write_path)

                disk_io_ms = (time.perf_counter() - t_io_start) * 1000.0
                os_metrics = self.monitor.snapshot_end(os_snap_start)
                return {
                    "status": "success",
                    "operation": op,
                    "size_label": size_label,
                    "file_format": file_format,
                    "rows_affected": len(df),
                    "columns_affected": len(df.columns),
                    "file_size_bytes": file_size_bytes,
                    "server_disk_io_ms": round(disk_io_ms, 3),
                    **os_metrics
                }
            else:
                return {"status": "error", "message": f"Operación desconocida: {op}"}

            disk_io_ms = (time.perf_counter() - t_io_start) * 1000.0
            file_size_bytes = os.path.getsize(filepath) if os.path.exists(filepath) else 0
            os_metrics = self.monitor.snapshot_end(os_snap_start)

            # Convertir el resultado a payload transferible (misma carga lógica entre formatos)
            t_ser0 = time.perf_counter()
            records_payload = df.to_dict(orient="list")
            server_ser_ms = (time.perf_counter() - t_ser0) * 1000.0

            return {
                "status": "success",
                "operation": op,
                "size_label": size_label,
                "file_format": file_format,
                "rows_affected": len(df),
                "columns_affected": len(df.columns),
                "file_size_bytes": file_size_bytes,
                "server_disk_io_ms": round(disk_io_ms, 3),
                "server_serialization_ms": round(server_ser_ms, 3),
                "data": records_payload,
                **os_metrics
            }

        except Exception as e:
            logger.error(f"Error ejecutando {op} ({file_format}): {e}")
            self.fault_tolerance.log_failure("IO_OPERATION_ERROR", "RaspberryPi", str(e))
            return {"status": "error", "message": str(e)}


# =====================================================================
# 1. Endpoint TCP Sockets (Reutiliza el framing de 4 bytes de protocol.py)
# =====================================================================
class TCPFileServer:
    def __init__(self, service: DistributedFileService, host: str = "0.0.0.0", port: int = 5000):
        self.service = service
        self.host = host
        self.port = port
        self.sock: Optional[socket.socket] = None
        self.is_running = False

    def start_background(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.listen(32)
        self.is_running = True
        t = threading.Thread(target=self._accept_loop, daemon=True)
        t.start()
        logger.info(f"[TCP File Server] Escuchando en {self.host}:{self.port}")

    def _accept_loop(self):
        while self.is_running:
            try:
                client_sock, addr = self.sock.accept()
                t = threading.Thread(target=self._handle_client, args=(client_sock, addr), daemon=True)
                t.start()
            except Exception:
                if self.is_running:
                    break

    def _handle_client(self, client_sock: socket.socket, addr: tuple):
        try:
            while self.is_running:
                frame = recv_frame(client_sock)
                if frame is None:
                    break
                req = json.loads(frame.decode("utf-8"))
                try:
                    resp = self.service.execute_operation(req)
                except ConnectionError as ce:
                    # Simular caída cerrando el socket abruptamente
                    logger.warning(f"[TCP] Cerrando conexión con {addr} por fallo simulado: {ce}")
                    client_sock.close()
                    return
                resp_bytes = json.dumps(resp).encode("utf-8")
                send_frame(client_sock, resp_bytes)
        except Exception:
            pass
        finally:
            try:
                client_sock.close()
            except Exception:
                pass

    def stop(self):
        self.is_running = False
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass


# =====================================================================
# 2. Endpoint gRPC (Genérico HTTP/2 sin requerir compilación externa)
# =====================================================================
class GRPCFileServer:
    def __init__(self, service: DistributedFileService, host: str = "0.0.0.0", port: int = 50051):
        self.service = service
        self.host = host
        self.port = port
        self.server = None

    def start_background(self):
        if not HAS_GRPC:
            logger.warning("[gRPC] Librería 'grpcio' no instalada. Endpoint gRPC deshabilitado.")
            return

        def handle_rpc(request_bytes: bytes, context) -> bytes:
            req = json.loads(request_bytes.decode("utf-8"))
            try:
                resp = self.service.execute_operation(req)
            except ConnectionError as ce:
                context.abort(grpc.StatusCode.UNAVAILABLE, str(ce))
                return b"{}"
            return json.dumps(resp).encode("utf-8")

        rpc_method_handlers = {
            "ExecuteOperation": grpc.unary_unary_rpc_method_handler(
                handle_rpc,
                request_deserializer=lambda x: x,
                response_serializer=lambda x: x,
            )
        }
        generic_handler = grpc.method_handlers_generic_handler(
            "storage.FileService", rpc_method_handlers
        )
        self.server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=16),
            options=[
                ("grpc.max_send_message_length", 64 * 1024 * 1024),
                ("grpc.max_receive_message_length", 64 * 1024 * 1024),
            ],
        )
        self.server.add_generic_rpc_handlers((generic_handler,))
        self.server.add_insecure_port(f"{self.host}:{self.port}")
        self.server.start()
        logger.info(f"[gRPC File Server] Escuchando en {self.host}:{self.port}")

    def stop(self):
        if self.server:
            self.server.stop(grace=0)


def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi File & I/O Service (TCP + gRPC)")
    parser.add_argument("--host", default="0.0.0.0", help="IP de escucha (default: 0.0.0.0)")
    parser.add_argument("--tcp-port", type=int, default=5000, help="Puerto TCP (default: 5000)")
    parser.add_argument("--grpc-port", type=int, default=50051, help="Puerto gRPC (default: 50051)")
    parser.add_argument("--storage-dir", default=STORAGE_DIR, help="Directorio de datasets CSV/Parquet")
    args = parser.parse_args()

    file_service = DistributedFileService(storage_dir=args.storage_dir)
    tcp_server = TCPFileServer(file_service, host=args.host, port=args.tcp_port)
    grpc_server = GRPCFileServer(file_service, host=args.host, port=args.grpc_port)

    tcp_server.start_background()
    grpc_server.start_background()

    logger.info("=== RASPBERRY PI FILE / I/O SERVICE ACTIVO ===")
    logger.info("Presione Ctrl+C para detener el servicio.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("Deteniendo servicios...")
        tcp_server.stop()
        grpc_server.stop()


if __name__ == "__main__":
    main()
