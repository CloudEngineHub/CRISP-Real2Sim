"""Runtime shims for CRISP helper wrappers.

This module is imported automatically by Python when its directory is added to
``PYTHONPATH``. We only patch torch.hub enough to make cached GitHub repos work
when GitHub branch probing fails (for example intermittent 502s).
"""

from __future__ import annotations

import os
from urllib.error import HTTPError, URLError


def _patch_torch_hub() -> None:
    try:
        import torch.hub as hub
    except Exception:
        return

    original = getattr(hub, "_parse_repo_info", None)
    if original is None or getattr(original, "__crisp_patched__", False):
        return

    def _parse_repo_info_with_cache_fallback(github: str):
        try:
            return original(github)
        except HTTPError as exc:
            if exc.code == 404:
                raise
        except URLError:
            pass
        except Exception:
            raise

        if ":" in github:
            repo_info, ref = github.split(":", 1)
        else:
            repo_info, ref = github, None

        repo_owner, repo_name = repo_info.split("/", 1)
        if ref is not None:
            return repo_owner, repo_name, ref

        hub_dir = hub.get_dir()
        for possible_ref in ("main", "master"):
            candidate = os.path.join(hub_dir, f"{repo_owner}_{repo_name}_{possible_ref}")
            if os.path.exists(candidate):
                return repo_owner, repo_name, possible_ref

        return original(github)

    _parse_repo_info_with_cache_fallback.__crisp_patched__ = True
    hub._parse_repo_info = _parse_repo_info_with_cache_fallback


_patch_torch_hub()
