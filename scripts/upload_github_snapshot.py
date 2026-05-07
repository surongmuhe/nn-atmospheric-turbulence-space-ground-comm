"""Upload the curated publishing snapshot to a private GitHub repository.

This script avoids requiring a local git executable. It uses the GitHub CLI
only to obtain the logged-in token, then uses the GitHub Git Data API to create
a single commit containing the snapshot.
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT = ROOT / "_github_publish"
GH = r"C:\Program Files\GitHub CLI\gh.exe"
API = "https://api.github.com"


def gh_token() -> str:
    token = subprocess.check_output([GH, "auth", "token"], text=True).strip()
    if not token:
        raise RuntimeError("GitHub CLI did not return an auth token.")
    return token


class GitHubClient:
    def __init__(self, token: str):
        self.token = token

    def request(self, method: str, path: str, payload: dict | None = None) -> dict:
        data = None
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "codex-thesis-publisher",
        }
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(
            API + path,
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = resp.read()
                return json.loads(body.decode("utf-8")) if body else {}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"GitHub API {method} {path} failed: {exc.code} {detail}") from exc

    def try_request(self, method: str, path: str) -> dict | None:
        req = urllib.request.Request(
            API + path,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "codex-thesis-publisher",
            },
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code in {404, 409}:
                return None
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"GitHub API {method} {path} failed: {exc.code} {detail}") from exc


def collect_files(snapshot: Path) -> list[Path]:
    files = sorted(path for path in snapshot.rglob("*") if path.is_file())
    too_large = [path for path in files if path.stat().st_size >= 100 * 1024 * 1024]
    if too_large:
        raise RuntimeError(
            "GitHub blocks files >=100MB without LFS: "
            + ", ".join(str(path.relative_to(snapshot)) for path in too_large)
        )
    return files


def unique_repo_name(client: GitHubClient, owner: str, base: str, reuse_existing: bool) -> str:
    existing = client.try_request("GET", f"/repos/{owner}/{base}")
    if existing is None or reuse_existing:
        return base
    suffix = time.strftime("%Y%m%d-%H%M%S")
    candidate = f"{base}-{suffix}"
    while client.try_request("GET", f"/repos/{owner}/{candidate}") is not None:
        time.sleep(1)
        suffix = time.strftime("%Y%m%d-%H%M%S")
        candidate = f"{base}-{suffix}"
    return candidate


def create_or_get_repo(client: GitHubClient, owner: str, repo: str, description: str) -> dict:
    existing = client.try_request("GET", f"/repos/{owner}/{repo}")
    if existing is not None:
        return existing
    return client.request(
        "POST",
        "/user/repos",
        {
            "name": repo,
            "description": description,
            "private": True,
            "auto_init": False,
            "has_issues": True,
            "has_projects": False,
            "has_wiki": False,
        },
    )


def upload_snapshot(client: GitHubClient, owner: str, repo: str, snapshot: Path) -> dict:
    base_commit_sha, base_tree_sha = ensure_initial_commit(client, owner, repo)
    files = collect_files(snapshot)
    tree_entries = []
    total = len(files)
    for idx, path in enumerate(files, start=1):
        rel = path.relative_to(snapshot).as_posix()
        content = base64.b64encode(path.read_bytes()).decode("ascii")
        blob = client.request(
            "POST",
            f"/repos/{owner}/{repo}/git/blobs",
            {"content": content, "encoding": "base64"},
        )
        tree_entries.append(
            {
                "path": rel,
                "mode": "100644",
                "type": "blob",
                "sha": blob["sha"],
            }
        )
        if idx % 50 == 0 or idx == total:
            print(f"uploaded blobs: {idx}/{total}")

    tree = client.request(
        "POST",
        f"/repos/{owner}/{repo}/git/trees",
        {
            "base_tree": base_tree_sha,
            "tree": [
                {"path": ".github_init", "mode": "100644", "type": "blob", "sha": None},
                *tree_entries,
            ],
        },
    )
    commit = client.request(
        "POST",
        f"/repos/{owner}/{repo}/git/commits",
        {
            "message": "Initial private thesis project snapshot",
            "tree": tree["sha"],
            "parents": [base_commit_sha],
        },
    )
    client.request(
        "PATCH",
        f"/repos/{owner}/{repo}/git/refs/heads/main",
        {"sha": commit["sha"], "force": False},
    )
    client.request(
        "PATCH",
        f"/repos/{owner}/{repo}",
        {"default_branch": "main"},
    )
    return {
        "file_count": total,
        "commit_sha": commit["sha"],
        "repo_url": f"https://github.com/{owner}/{repo}",
    }


def ensure_initial_commit(client: GitHubClient, owner: str, repo: str) -> tuple[str, str]:
    ref = client.try_request("GET", f"/repos/{owner}/{repo}/git/ref/heads/main")
    if ref is None:
        client.request(
            "PUT",
            f"/repos/{owner}/{repo}/contents/.github_init",
            {
                "message": "Initialize repository",
                "content": base64.b64encode(b"initialized\n").decode("ascii"),
                "branch": "main",
            },
        )
        ref = client.request("GET", f"/repos/{owner}/{repo}/git/ref/heads/main")
    commit_sha = ref["object"]["sha"]
    commit = client.request("GET", f"/repos/{owner}/{repo}/git/commits/{commit_sha}")
    return commit_sha, commit["tree"]["sha"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--repo", default="turbulence-lstm-satcom-thesis")
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()

    snapshot = args.snapshot.resolve()
    if not snapshot.exists():
        raise FileNotFoundError(snapshot)

    token = gh_token()
    client = GitHubClient(token)
    user = client.request("GET", "/user")
    owner = user["login"]
    repo = unique_repo_name(client, owner, args.repo, args.reuse_existing)
    repo_info = create_or_get_repo(
        client,
        owner,
        repo,
        "Private thesis project: neural-network atmospheric turbulence forecasting and satellite-to-ground optical communication.",
    )
    result = upload_snapshot(client, owner, repo, snapshot)
    result["private"] = repo_info.get("private", True)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
