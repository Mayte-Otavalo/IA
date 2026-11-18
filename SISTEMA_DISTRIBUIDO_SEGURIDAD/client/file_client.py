"""
Cliente Coordinador y de Pruebas (COURSE LINUX VM / PC TEST CLIENT).
Implementa las funciones de las Diapositivas 3, 4, 5, 6 y 9:
- Operaciones locales de referencia (Local Baseline).
- Peticiones remotas a la Raspberry Pi mediante Sockets TCP y gRPC.
- Separación explícita entre tiempo de Disco I/O, costo de serialización y overhead de red.
- Manejo de fallos: Sin reintento (No retry) vs. Timeout + Reintento (Timeout + retry).
"""

import os
import sys
import time
import json
import socket
import logging
from typing import Dict, Any, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from protocol import send_frame, recv_frame
from server.file_server import DistributedFileService, OSResourceMonitor, HAS_GRPC
from prepare_datasets import STORAGE_DIR

if HAS_GRPC:
    import grpc

logger = logging.getLogger("FileTestClient")


class DistributedStorageClient:
    """Cliente de pruebas para evaluación de almacenamiento local y remoto (TCP / gRPC)."""

    def __init__(
        self,
        pi_host: str = "127.0.0.1",
        tcp_port: int = 5000,
        grpc_port: int = 50051,
        timeout_sec: float = 5.0,
        local_storage_dir: str = STORAGE_DIR
    ):
        self.pi_host = pi_host
        self.tcp_port = tcp_port
        self.grpc_port = grpc_port
        self.timeout_sec = timeout_sec
        self.local_service = DistributedFileService(storage_dir=local_storage_dir)
        self.client_monitor = OSResourceMonitor()

    def execute_local_baseline(self, req: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecuta la operación directamente en el almacenamiento local de la PC/Contenedor (Baseline)."""
        snap0 = self.client_monitor.snapshot_start()
        t0 = time.perf_counter()
        resp = self.local_service.execute_operation(req)
        total_wall_ms = (time.perf_counter() - t0) * 1000.0
        os_client = self.client_monitor.snapshot_end(snap0)

        disk_io_ms = resp.get("server_disk_io_ms", total_wall_ms)
        ser_ms = resp.get("server_serialization_ms", 0.0)
        file_bytes = resp.get("file_size_bytes", 0)
        throughput_mbps = (file_bytes / (1024.0 * 1024.0)) / (total_wall_ms / 1000.0) if total_wall_ms > 0 else 0.0

        return {
            "status": resp.get("status", "error"),
            "mode": "local",
            "transport": "local_fs",
            "operation": req.get("operation"),
            "size_label": req.get("size_label", "small"),
            "file_format": req.get("file_format", "csv"),
            "rows_affected": resp.get("rows_affected", 0),
            "file_size_bytes": file_bytes,
            "bytes_transmitted": 0,
            "total_wall_time_ms": round(total_wall_ms, 3),
            "disk_io_time_ms": round(disk_io_ms, 3),
            "serialization_time_ms": round(ser_ms, 3),
            "network_overhead_ms": 0.0,
            "throughput_mb_s": round(throughput_mbps, 3),
            "client_cpu_percent": os_client["server_cpu_percent"],
            "client_ram_mb": os_client["server_ram_mb"],
            "server_cpu_percent": resp.get("server_cpu_percent", 0.0),
            "server_ram_mb": resp.get("server_ram_mb", 0.0),
            "os_disk_read_bytes": resp.get("os_disk_read_bytes", 0),
            "os_disk_write_bytes": resp.get("os_disk_write_bytes", 0),
            "failed_attempts": 0,
            "detection_time_ms": 0.0,
            "recovery_time_ms": 0.0
        }

    def _send_tcp_once(self, req: Dict[str, Any]) -> Dict[str, Any]:
        """Envía una petición por socket TCP a la Raspberry Pi y desglosa los tiempos."""
        snap0 = self.client_monitor.snapshot_start()
        t_wall0 = time.perf_counter()

        t_ser0 = time.perf_counter()
        req_bytes = json.dumps(req).encode("utf-8")
        client_ser_ms = (time.perf_counter() - t_ser0) * 1000.0

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.timeout_sec)
        try:
            sock.connect((self.pi_host, self.tcp_port))
            send_frame(sock, req_bytes)
            resp_bytes = recv_frame(sock)
            if resp_bytes is None:
                raise ConnectionResetError("El servidor de la Raspberry Pi cerró la conexión sin responder")
        finally:
            try:
                sock.close()
            except Exception:
                pass

        t_des0 = time.perf_counter()
        resp = json.loads(resp_bytes.decode("utf-8"))
        client_des_ms = (time.perf_counter() - t_des0) * 1000.0

        total_wall_ms = (time.perf_counter() - t_wall0) * 1000.0
        os_client = self.client_monitor.snapshot_end(snap0)

        if resp.get("status") != "success":
            raise RuntimeError(resp.get("message", "Error en servidor remoto"))

        server_disk_ms = resp.get("server_disk_io_ms", 0.0)
        server_ser_ms = resp.get("server_serialization_ms", 0.0)
        total_ser_ms = client_ser_ms + client_des_ms + server_ser_ms
        net_overhead_ms = max(0.0, total_wall_ms - server_disk_ms - total_ser_ms)

        bytes_tx = len(req_bytes) + len(resp_bytes) + 8  # Incluye los headers de 4 bytes
        throughput_mbps = (bytes_tx / (1024.0 * 1024.0)) / (total_wall_ms / 1000.0) if total_wall_ms > 0 else 0.0

        return {
            "status": "success",
            "mode": "remote",
            "transport": "tcp",
            "operation": req.get("operation"),
            "size_label": req.get("size_label", "small"),
            "file_format": req.get("file_format", "csv"),
            "rows_affected": resp.get("rows_affected", 0),
            "file_size_bytes": resp.get("file_size_bytes", 0),
            "bytes_transmitted": bytes_tx,
            "total_wall_time_ms": round(total_wall_ms, 3),
            "disk_io_time_ms": round(server_disk_ms, 3),
            "serialization_time_ms": round(total_ser_ms, 3),
            "network_overhead_ms": round(net_overhead_ms, 3),
            "throughput_mb_s": round(throughput_mbps, 3),
            "client_cpu_percent": os_client["server_cpu_percent"],
            "client_ram_mb": os_client["server_ram_mb"],
            "server_cpu_percent": resp.get("server_cpu_percent", 0.0),
            "server_ram_mb": resp.get("server_ram_mb", 0.0),
            "os_disk_read_bytes": resp.get("os_disk_read_bytes", 0),
            "os_disk_write_bytes": resp.get("os_disk_write_bytes", 0),
        }

    def _send_grpc_once(self, req: Dict[str, Any]) -> Dict[str, Any]:
        """Envía una petición por gRPC a la Raspberry Pi y desglosa los tiempos."""
        if not HAS_GRPC:
            raise RuntimeError("grpcio no está instalado en este entorno")

        snap0 = self.client_monitor.snapshot_start()
        t_wall0 = time.perf_counter()

        t_ser0 = time.perf_counter()
        req_bytes = json.dumps(req).encode("utf-8")
        client_ser_ms = (time.perf_counter() - t_ser0) * 1000.0

        channel = grpc.insecure_channel(
            f"{self.pi_host}:{self.grpc_port}",
            options=[
                ("grpc.max_send_message_length", 64 * 1024 * 1024),
                ("grpc.max_receive_message_length", 64 * 1024 * 1024),
            ],
        )
        try:
            call = channel.unary_unary(
                "/storage.FileService/ExecuteOperation",
                request_serializer=lambda x: x,
                response_deserializer=lambda x: x,
            )
            resp_bytes = call(req_bytes, timeout=self.timeout_sec)
        finally:
            channel.close()

        t_des0 = time.perf_counter()
        resp = json.loads(resp_bytes.decode("utf-8"))
        client_des_ms = (time.perf_counter() - t_des0) * 1000.0

        total_wall_ms = (time.perf_counter() - t_wall0) * 1000.0
        os_client = self.client_monitor.snapshot_end(snap0)

        if resp.get("status") != "success":
            raise RuntimeError(resp.get("message", "Error en servidor gRPC remoto"))

        server_disk_ms = resp.get("server_disk_io_ms", 0.0)
        server_ser_ms = resp.get("server_serialization_ms", 0.0)
        total_ser_ms = client_ser_ms + client_des_ms + server_ser_ms
        net_overhead_ms = max(0.0, total_wall_ms - server_disk_ms - total_ser_ms)

        bytes_tx = len(req_bytes) + len(resp_bytes)
        throughput_mbps = (bytes_tx / (1024.0 * 1024.0)) / (total_wall_ms / 1000.0) if total_wall_ms > 0 else 0.0

        return {
            "status": "success",
            "mode": "remote",
            "transport": "grpc",
            "operation": req.get("operation"),
            "size_label": req.get("size_label", "small"),
            "file_format": req.get("file_format", "csv"),
            "rows_affected": resp.get("rows_affected", 0),
            "file_size_bytes": resp.get("file_size_bytes", 0),
            "bytes_transmitted": bytes_tx,
            "total_wall_time_ms": round(total_wall_ms, 3),
            "disk_io_time_ms": round(server_disk_ms, 3),
            "serialization_time_ms": round(total_ser_ms, 3),
            "network_overhead_ms": round(net_overhead_ms, 3),
            "throughput_mb_s": round(throughput_mbps, 3),
            "client_cpu_percent": os_client["server_cpu_percent"],
            "client_ram_mb": os_client["server_ram_mb"],
            "server_cpu_percent": resp.get("server_cpu_percent", 0.0),
            "server_ram_mb": resp.get("server_ram_mb", 0.0),
            "os_disk_read_bytes": resp.get("os_disk_read_bytes", 0),
            "os_disk_write_bytes": resp.get("os_disk_write_bytes", 0),
        }

    def execute_remote(
        self,
        req: Dict[str, Any],
        transport: str = "tcp",
        retry_enabled: bool = True,
        max_retries: int = 6,
        retry_delay_sec: float = 0.8
    ) -> Dict[str, Any]:
        """
        Ejecuta una petición remota hacia la Raspberry Pi usando TCP o gRPC.
        Soporta modo 'No retry' (retry_enabled=False) vs 'Timeout + retry' (retry_enabled=True).
        Registra tiempo de detección de fallo, intentos fallidos y tiempo total de recuperación.
        """
        attempts = max_retries if retry_enabled else 1
        failed_attempts = 0
        detection_time_ms = 0.0
        t_first_failure = None

        for attempt in range(1, attempts + 1):
            t_attempt0 = time.perf_counter()
            try:
                if transport.lower() == "grpc":
                    result = self._send_grpc_once(req)
                else:
                    result = self._send_tcp_once(req)

                recovery_ms = (time.perf_counter() - t_first_failure) * 1000.0 if t_first_failure else 0.0
                result["failed_attempts"] = failed_attempts
                result["detection_time_ms"] = round(detection_time_ms, 3)
                result["recovery_time_ms"] = round(recovery_ms, 3)
                return result

            except Exception as e:
                err_elapsed_ms = (time.perf_counter() - t_attempt0) * 1000.0
                failed_attempts += 1
                if t_first_failure is None:
                    t_first_failure = t_attempt0
                    detection_time_ms = err_elapsed_ms

                if not retry_enabled or attempt == attempts:
                    total_elapsed = (time.perf_counter() - t_first_failure) * 1000.0
                    return {
                        "status": "failed",
                        "error": str(e),
                        "mode": "remote",
                        "transport": transport,
                        "operation": req.get("operation"),
                        "size_label": req.get("size_label", "small"),
                        "file_format": req.get("file_format", "csv"),
                        "rows_affected": 0,
                        "file_size_bytes": 0,
                        "bytes_transmitted": 0,
                        "total_wall_time_ms": round(total_elapsed, 3),
                        "disk_io_time_ms": 0.0,
                        "serialization_time_ms": 0.0,
                        "network_overhead_ms": 0.0,
                        "throughput_mb_s": 0.0,
                        "client_cpu_percent": 0.0,
                        "client_ram_mb": 0.0,
                        "server_cpu_percent": 0.0,
                        "server_ram_mb": 0.0,
                        "os_disk_read_bytes": 0,
                        "os_disk_write_bytes": 0,
                        "failed_attempts": failed_attempts,
                        "detection_time_ms": round(detection_time_ms, 3),
                        "recovery_time_ms": 0.0
                    }

                time.sleep(retry_delay_sec)
