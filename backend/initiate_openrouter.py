import subprocess
import os
import time
import requests
from pathlib import Path

def start_opencode_server(
    port: int = 4096,
    hostname: str = "127.0.0.1",
    password: str | None = None,
    log_file: str = "opencode_server.log"
) -> subprocess.Popen:
    try:
        resp = requests.get(f"http://{hostname}:{port}/global/health", timeout=2)
        if resp.status_code == 200:
            print(f"✅ OpenCode server already running on http://{hostname}:{port}")
            return None
    except requests.exceptions.RequestException:
        pass

    env = os.environ.copy()
    if password:
        env["OPENCODE_SERVER_PASSWORD"] = password

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    log_handle = open(log_path, "a")

    process = subprocess.Popen(
        ["opencode", "serve", "--port", str(port), "--hostname", hostname],
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        shell=True
    )

    print(f"🚀 Starting OpenCode server on http://{hostname}:{port} (PID: {process.pid})")

    for _ in range(15):
        try:
            resp = requests.get(f"http://{hostname}:{port}/global/health", timeout=2)
            if resp.status_code == 200:
                print("✅ OpenCode server is ready!")
                return process
        except requests.exceptions.RequestException:
            time.sleep(1)

    print("⚠️ Server started but health check failed. Check logs.")
    return process

def kill_opencode_server(process: subprocess.Popen | None = None, port: int = 4096):
    """
    Kills the OpenCode server.
    - If process is provided → kill directly
    - Else → find process using the port and kill it
    """

    # Case 1: You have process handle
    if process:
        try:
            process.terminate()
            process.wait(timeout=5)
            print(f"🛑 Killed OpenCode server (PID: {process.pid})")
            return
        except Exception:
            process.kill()
            print(f"⚠️ Force killed OpenCode server (PID: {process.pid})")
            return

    # Case 2: Kill by port (Windows)
    try:
        result = subprocess.check_output(
            f'netstat -ano | findstr :{port}',
            shell=True
        ).decode()

        lines = result.strip().split("\n")
        pids = set()

        for line in lines:
            parts = line.split()
            if len(parts) >= 5:
                pids.add(parts[-1])

        for pid in pids:
            subprocess.run(f"taskkill /PID {pid} /F", shell=True)
            print(f"🛑 Killed process on port {port} (PID: {pid})")

        if not pids:
            print(f"ℹ️ No process found on port {port}")

    except subprocess.CalledProcessError:
        print(f"ℹ️ No process found on port {port}")
if __name__ == "__main__":
    server_process = start_opencode_server(port=4096)