"""Locate the ``guoq`` and ``queso`` binaries installed with this package.

The wheel installs the Rust binaries as console scripts, not as package data, so
importing this package tells you nothing about where they are. These helpers search the
places pip installs scripts to — the same dance ruff does for its binary — so callers
(such as wisq) can spawn the optimizer without caring how it was installed.
"""

import os
import shutil
import sys
import sysconfig

__all__ = ["find_guoq_bin", "find_queso_bin", "find_bqskit_worker"]


def _find_bin(name):
    exe = name + (".exe" if os.name == "nt" else "")

    path = os.path.join(sysconfig.get_path("scripts"), exe)
    if os.path.isfile(path):
        return path

    # User-scheme installs (`pip install --user`) put scripts elsewhere.
    if sys.version_info >= (3, 10):
        user_scheme = sysconfig.get_preferred_scheme("user")
    elif os.name == "nt":
        user_scheme = "nt_user"
    elif sys.platform == "darwin" and getattr(sys, "_framework", None):
        user_scheme = "osx_framework_user"
    else:
        user_scheme = "posix_user"
    path = os.path.join(sysconfig.get_path("scripts", scheme=user_scheme), exe)
    if os.path.isfile(path):
        return path

    path = shutil.which(exe)
    if path is not None:
        return path

    raise FileNotFoundError(
        f"could not find the {name!r} binary; expected it to be installed as a "
        f"console script alongside the `guoq` Python package"
    )


def find_guoq_bin():
    """Absolute path to the ``guoq`` optimizer binary."""
    return _find_bin("guoq")


def find_queso_bin():
    """Absolute path to the ``queso`` rule-synthesis binary."""
    return _find_bin("queso")


def find_bqskit_worker():
    """Absolute path to the BQSKit worker script bundled with this package.

    Pass it to ``guoq --bqskit-worker``; guoq spawns and owns the worker process. The
    canonical copy lives at py/bqskit_worker.py in the repository, and CI checks that
    the two stay identical.
    """
    path = os.path.join(os.path.dirname(__file__), "bqskit_worker.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"bqskit_worker.py missing from installed package: {path}")
    return path
