# -*- coding: utf-8 -*-
"""Jupyter/Colab notebook helper for resolving the project root consistently."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

DEFAULT_REPO_URL = "https://github.com/liamyoum/W14-gpt-lab.git"
DEFAULT_REPO_BRANCH = "WEI"
DEFAULT_REPO_NAME = "W14-gpt-lab"


def normalize_github_url(url: str) -> str:
    """Normalize user input into a cloneable https GitHub URL."""
    url = url.strip()
    if not url:
        raise ValueError("GitHub 저장소 URL을 입력해야 합니다.")
    if url.startswith("github.com/"):
        url = "https://" + url
    if not url.startswith("https://"):
        raise ValueError("저장소 URL은 https://github.com/... 또는 github.com/... 형식이어야 합니다.")
    url = url.rstrip("/")
    if not url.endswith(".git"):
        url += ".git"
    return url


def is_repo_root(path: Path) -> bool:
    """Return True when the path looks like the project root."""
    return (
        path.exists()
        and path.is_dir()
        and (path / "src").is_dir()
        and (path / "scripts" / "run_finetune_light.sh").exists()
        and (path / "gpt-lab.ipynb").exists()
    )


def find_repo_dir(start: Path | None = None) -> Path | None:
    """Try to locate the repo from cwd, parents, and common Colab locations."""
    if start is None:
        start = Path.cwd().resolve()

    search_roots = [start, *start.parents]
    for root in search_roots:
        if is_repo_root(root):
            return root
        candidate = root / DEFAULT_REPO_NAME
        if is_repo_root(candidate):
            return candidate

    content_default = Path("/content") / DEFAULT_REPO_NAME
    if is_repo_root(content_default):
        return content_default

    content_dir = Path("/content")
    if content_dir.exists():
        for child in content_dir.iterdir():
            if is_repo_root(child):
                return child

    return None


def clone_repo(repo_url: str, token: str = "", branch: str = DEFAULT_REPO_BRANCH) -> Path:
    """Clone the repo into /content when running in Colab and it is missing."""
    repo_url = normalize_github_url(repo_url)
    clone_url = repo_url.replace("https://", f"https://{token}@") if token else repo_url
    repo_name = Path(repo_url[:-4]).name if repo_url.endswith(".git") else Path(repo_url).name
    repo_dir = Path("/content") / repo_name

    if not repo_dir.exists():
        cmd = ["git", "clone"]
        if branch:
            cmd += ["-b", branch]
        cmd += [clone_url, str(repo_dir)]
        subprocess.run(cmd, check=True)
        subprocess.run(["git", "remote", "set-url", "origin", repo_url], cwd=repo_dir, check=True)
    return repo_dir


def setup_repo_dir(
    repo_url: str | None = None,
    token: str = "",
    branch: str = DEFAULT_REPO_BRANCH,
    clone_if_missing: bool = True,
) -> Path:
    """Resolve repo_dir, optionally cloning in Colab, and add src to sys.path."""
    repo_dir = find_repo_dir()
    if repo_dir is None and "google.colab" in sys.modules and clone_if_missing:
        repo_dir = clone_repo(repo_url or DEFAULT_REPO_URL, token=token, branch=branch)
    if repo_dir is None:
        raise FileNotFoundError(
            "W14-gpt-lab 경로를 찾지 못했습니다. "
            "Colab에서는 맨 위 설정 셀을 먼저 실행해 저장소를 clone하세요."
        )

    os.chdir(repo_dir)
    src_path = str(repo_dir / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    return repo_dir
