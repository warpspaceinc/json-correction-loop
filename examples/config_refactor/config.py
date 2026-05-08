"""Clean and broken config fixtures."""
from __future__ import annotations

import copy
from typing import Any


CLEAN: dict[str, Any] = {
    "app_name": "warpspace-api",
    "version": "1.4.2",
    "services": {
        "web": {
            "image": "warpspace/web:1.4.2",
            "port": 8080,
            "replicas": 3,
            "env": "prod",
        },
        "worker": {
            "image": "warpspace/worker:1.4.2",
            "port": 9090,
            "replicas": 2,
            "env": "prod",
        },
    },
}


# Deliberately broken in three ways the schema critic detects:
#   1. version doesn't match X.Y.Z (typo "1.4")
#   2. web.port is a string ("8080") instead of integer
#   3. worker.env is "production" — not in the enum
BROKEN: dict[str, Any] = {
    "app_name": "warpspace-api",
    "version": "1.4",
    "services": {
        "web": {
            "image": "warpspace/web:1.4.2",
            "port": "8080",
            "replicas": 3,
            "env": "prod",
        },
        "worker": {
            "image": "warpspace/worker:1.4.2",
            "port": 9090,
            "replicas": 2,
            "env": "production",
        },
    },
}


def clean() -> dict[str, Any]:
    return copy.deepcopy(CLEAN)


def broken() -> dict[str, Any]:
    return copy.deepcopy(BROKEN)
