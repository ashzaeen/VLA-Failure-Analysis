from __future__ import annotations

import argparse

import uvicorn

from src.server.app import create_app
from src.server.configs import ServerConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/server/pi0_libero_finetuned_v044_server.yaml')
    args = parser.parse_args()

    config = ServerConfig.from_yaml(args.config)
    app = create_app(config)
    uvicorn.run(app, host=config.server.host, port=config.server.port)


if __name__ == '__main__':
    main()
