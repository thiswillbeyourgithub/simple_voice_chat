import socket
import sys
import time
import errno
from loguru import logger


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Check if a given port is already in use on the specified host."""
    time.sleep(0.1)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if hasattr(socket, "SO_REUSEADDR"):
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if sys.platform == "win32" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            s.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)

        s.bind((host, port))
        s.close()
        return False
    except socket.error as e:
        if e.errno in [
            errno.EADDRINUSE,
            getattr(errno, "WSAEADDRINUSE", None),
        ]:  # Check both Linux/Windows codes
            logger.debug(f"Port {port} on {host} is in use ({e.strerror}).")
            return True
        else:
            logger.error(f"Unexpected socket error checking port {port}: {e}")
            return True
    except Exception as e:
        logger.error(f"Unexpected error checking port {port}: {e}")
        return True
    finally:
        s.close()
