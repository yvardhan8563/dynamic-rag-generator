"""Start both local servers, probe them, and clean up. Does not call an LLM."""
import os
from pathlib import Path
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

import httpx

ROOT = Path(__file__).resolve().parents[1]


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_for(url, process):
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Server exited before becoming ready")
        try:
            response = httpx.get(url, timeout=2, trust_env=False)
            if response.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.25)
    raise RuntimeError("Server did not become ready within 45 seconds")


def main():
    backend_port, frontend_port = free_port(), free_port()
    processes = []
    with TemporaryDirectory(prefix="rag-server-smoke-") as data_dir:
        env = {**os.environ, "RAG_DATA_DIR": data_dir,
               "RAG_API_URL": f"http://127.0.0.1:{backend_port}"}
        commands = [
            [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(backend_port)],
            [sys.executable, "-m", "streamlit", "run", "frontend/streamlit_app.py", "--server.headless", "true",
             "--server.port", str(frontend_port)],
        ]
        try:
            for command in commands:
                processes.append(subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0))
            wait_for(f"http://127.0.0.1:{backend_port}/api/v1/health/live", processes[0])
            wait_for(f"http://127.0.0.1:{backend_port}/docs", processes[0])
            wait_for(f"http://127.0.0.1:{frontend_port}/_stcore/health", processes[1])
            wait_for(f"http://127.0.0.1:{frontend_port}/", processes[1])
            print("PASS: FastAPI liveness/docs and Streamlit health/root respond over HTTP.")
        finally:
            for process in processes:
                if process.poll() is None:
                    if os.name == "nt":
                        # Windows venv python can be a launcher with a child process.
                        # Stop only this script's process tree, before the parent exits.
                        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                       creationflags=subprocess.CREATE_NO_WINDOW, check=False)
                    else:
                        process.terminate()
            for process in processes:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


if __name__ == "__main__":
    main()
