"""Start DA3, DEVA, and TrackCraft from the Caster environment."""

import logging
import signal
import subprocess
import sys
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig
from termcolor import colored


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config/servers", config_name="servers")
def main(config: DictConfig) -> None:
    stopping = False

    def stop(*_: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    da3 = Path(sys.executable).parent / "da3"
    commands = {
        "DA3": [str(da3), "backend", "--port", str(config.da3_port)],
        "DEVA": [sys.executable, "-m", "caster.servers.mask_server"],
        "TrackCraft": [sys.executable, "-m", "caster.servers.trackcraft_server"],
    }
    processes = {}
    try:
        for name, command in commands.items():
            logger.info("%s %s", colored("Starting server:", "cyan"), name)
            processes[name] = subprocess.Popen(command, start_new_session=True)
        while not stopping:
            for name, process in processes.items():
                if process.poll() is not None:
                    raise RuntimeError(
                        f"{name} server exited with code {process.returncode}"
                    )
            time.sleep(2)
    finally:
        logger.info(colored("Stopping Caster servers", "yellow"))
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        for process in processes.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()
