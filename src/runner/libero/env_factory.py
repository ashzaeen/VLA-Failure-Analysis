from __future__ import annotations

import os
from typing import Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
except Exception:  # pragma: no cover
    get_libero_path = None
    OffScreenRenderEnv = None


class LiberoEnvFactory:
    def create(self, task, resolution: int = 256, seed: int = 7):
        if OffScreenRenderEnv is None or get_libero_path is None:
            raise ImportError('LIBERO is not installed in this environment.')
        task_description = task.language
        task_bddl_file = os.path.join(get_libero_path('bddl_files'), task.problem_folder, task.bddl_file)
        env_args = {
            'bddl_file_name': task_bddl_file,
            'camera_heights': resolution,
            'camera_widths': resolution,
        }
        env = OffScreenRenderEnv(**env_args)
        env.seed(seed)
        logger.info('Initialized LIBERO environment.')
        return env, task_description
