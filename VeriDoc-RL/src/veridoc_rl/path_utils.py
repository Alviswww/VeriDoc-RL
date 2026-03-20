from __future__ import annotations

import os


def expand_env_and_user(value: str) -> str:
    return os.path.expanduser(os.path.expandvars(value))
