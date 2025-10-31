import grp
import os
import socket
import logging

import uvicorn

import config
from backend.views import app


if __name__ == "__main__":
    if config.DEBUG:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        # Remove old socket (if any).
        try:
            os.unlink(config.SOCKET_FILE)
        except FileNotFoundError as e:
            logging.info(f"No old socket file found: {e}")

        # Create socket and run app.
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(config.SOCKET_FILE)

                os.chmod(config.SOCKET_FILE, 0o775)
                os.chown(config.SOCKET_FILE, -1, grp.getgrnam("www-data").gr_gid)

                uvicorn.run(app, uds=config.SOCKET_FILE, access_log=False)
            except OSError as e:
                logging.warning(e)
