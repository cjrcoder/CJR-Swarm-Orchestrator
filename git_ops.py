"""GitHub operations controller — clone, commit, and push via subprocess."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Final

from schemas import CodePayload, CommitInfo, GitPushResult, SwarmConfig

logger: logging.Logger = logging.getLogger(__name__)

_MAX_RETRIES: Final[int] = 3
_RETRY_BACKOFF_BASE: Final[float] = 2.0


class GitController:
    """Manages git operations for pushing generated code to target repositories."""

    def __init__(self, config: SwarmConfig) -> None:
        self._config: SwarmConfig = config
        self._pat: str = config.gh_pat.get_secret_value()
        self._owner: str = config.target_repo_owner
        logger.info("GitController initialised for owner=%s", self._owner)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_remote_url(self, repo_name: str) -> str:
        """Return an HTTPS remote URL with embedded PAT for authentication.

        Format: ``https://{pat}@github.com/{owner}/{repo_name}.git``
        """
        return f"https://{self._pat}@github.com/{self._owner}/{repo_name}.git"

    @staticmethod
    def _redact_pat(text: str) -> str:
        """Replace any PAT-like tokens in *text* so they never leak into logs."""
        return re.sub(
            r"https://[^@]+@github\.com",
            "https://***@github.com",
            text,
        )

    def _run_git(
        self,
        args: list[str],
        cwd: str,
        *,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        """Execute a ``git`` command in *cwd*.

        * Captures stdout/stderr as text.
        * Redacts PAT tokens from all log output.
        * Raises ``subprocess.CalledProcessError`` when *check* is True and
          the command exits non-zero.
        """
        cmd = ["git", *args]
        safe_cmd = self._redact_pat(" ".join(cmd))
        logger.debug("⚙️  git command: %s (cwd=%s)", safe_cmd, cwd)

        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

        if result.stdout:
            logger.debug("stdout: %s", self._redact_pat(result.stdout.strip()))
        if result.stderr:
            logger.debug("stderr: %s", self._redact_pat(result.stderr.strip()))

        if check and result.returncode != 0:
            safe_stderr = self._redact_pat(result.stderr.strip())
            raise subprocess.CalledProcessError(
                result.returncode,
                safe_cmd,
                output=result.stdout,
                stderr=safe_stderr,
            )

        return result

    # ------------------------------------------------------------------
    # Clone / init helpers
    # ------------------------------------------------------------------

    def _clone_or_init(self, remote_url: str, workdir: str) -> None:
        """Clone the remote repository, or ``git init`` if it does not exist."""
        try:
            self._run_git(["clone", remote_url, "."], cwd=workdir)
            logger.info("📦 Repository cloned successfully")
        except subprocess.CalledProcessError as exc:
            stderr_lower = (exc.stderr or "").lower()
            if "not found" in stderr_lower or "repository" in stderr_lower:
                logger.warning(
                    "⚠️  Remote repository not found — initialising new repo"
                )
                self._run_git(["init"], cwd=workdir)
                self._run_git(["remote", "add", "origin", remote_url], cwd=workdir)
                # Create an initial commit so the branch exists
                readme = Path(workdir) / "README.md"
                readme.write_text(
                    "# New Repository\n\nInitialised by CJR Swarm Orchestrator.\n",
                    encoding="utf-8",
                )
                self._run_git(["add", "."], cwd=workdir)
                self._run_git(
                    ["commit", "-m", "chore: initial commit"], cwd=workdir
                )
            elif "authentication" in stderr_lower or "403" in stderr_lower:
                raise RuntimeError(
                    "🔒 GitHub authentication failed — verify your PAT "
                    "has 'repo' scope and is not expired."
                ) from exc
            else:
                raise

    # ------------------------------------------------------------------
    # Write files
    # ------------------------------------------------------------------

    @staticmethod
    def _write_code_files(code: CodePayload, workdir: str) -> list[str]:
        """Write every ``CodeFile`` in *code.files* to the working directory.

        Returns a list of relative paths that were written.
        """
        written: list[str] = []
        for code_file in code.files:
            target = Path(workdir) / code_file.filepath
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(code_file.content, encoding="utf-8")
            written.append(code_file.filepath)
            logger.debug("   📝 wrote %s (%d bytes)", code_file.filepath, len(code_file.content))
        logger.info("📁 Wrote %d files to working directory", len(written))
        return written

    # ------------------------------------------------------------------
    # Commit message generation
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_commit_message(plan_name: str, files: list[str]) -> str:
        """Generate a Conventional Commit message.

        Format::

            feat(swarm): implement {plan_name}

            Files included:
            - path/to/file_a.py
            - path/to/file_b.py
        """
        subject = f"feat(swarm): implement {plan_name}"
        body_lines = ["", "Files included:"]
        body_lines.extend(f"- {f}" for f in sorted(files))
        return "\n".join([subject, *body_lines])

    # ------------------------------------------------------------------
    # Network-aware push with retry
    # ------------------------------------------------------------------

    def _push_with_retry(
        self,
        workdir: str,
        branch: str,
    ) -> None:
        """Push to origin with exponential-backoff retries on network errors."""
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                self._run_git(["push", "-u", "origin", branch], cwd=workdir)
                return
            except subprocess.CalledProcessError as exc:
                stderr_lower = (exc.stderr or "").lower()
                is_network = any(
                    tok in stderr_lower
                    for tok in ("could not resolve", "connection", "timed out", "ssl")
                )
                if is_network and attempt < _MAX_RETRIES:
                    wait = _RETRY_BACKOFF_BASE ** attempt
                    logger.warning(
                        "🌐 Network error on push attempt %d/%d — retrying in %.1fs",
                        attempt,
                        _MAX_RETRIES,
                        wait,
                    )
                    time.sleep(wait)
                else:
                    raise

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push_code(
        self,
        code: CodePayload,
        repo_name: str,
        commit_info: CommitInfo,
    ) -> GitPushResult:
        """Clone (or init), commit generated code, and push to GitHub.

        Parameters
        ----------
        code:
            The ``CodePayload`` produced by the implementer agent.
        repo_name:
            Target repository name under ``self._owner``.
        commit_info:
            Author, branch, and message metadata for the commit.

        Returns
        -------
        GitPushResult
            Success flag, commit SHA (if any), and the remote URL.
        """
        remote_url = self._build_remote_url(repo_name)
        safe_url = self._redact_pat(remote_url)
        tmpdir: str | None = None

        try:
            # 1. Create temp workspace
            tmpdir = tempfile.mkdtemp(prefix="cjr_swarm_")
            logger.info("🗂️  Temp workspace: %s", tmpdir)

            # 2. Clone or init
            self._clone_or_init(remote_url, tmpdir)

            # 3. Checkout / create target branch
            branch = commit_info.branch or "main"
            self._run_git(
                ["checkout", "-B", branch], cwd=tmpdir, check=True
            )

            # 4. Write generated files
            written = self._write_code_files(code, tmpdir)

            # 5. Configure author identity
            # CommitInfo.author is e.g. "CJR-Swarm-Agent <swarm@cjrcoder.dev>"
            raw_author = commit_info.author or "CJR Swarm Bot <swarm-bot@cjr.dev>"
            if "<" in raw_author and ">" in raw_author:
                author_name = raw_author.split("<")[0].strip()
                author_email = raw_author.split("<")[1].rstrip(">")
            else:
                author_name = raw_author
                author_email = "swarm-bot@cjr.dev"
            self._run_git(
                ["config", "user.name", author_name], cwd=tmpdir
            )
            self._run_git(
                ["config", "user.email", author_email], cwd=tmpdir
            )

            # 6. Stage all changes
            self._run_git(["add", "-A"], cwd=tmpdir)

            # 7. Commit
            message = commit_info.message or self._generate_commit_message(
                repo_name, written
            )
            self._run_git(["commit", "-m", message], cwd=tmpdir)

            # 8. Push with retry
            self._push_with_retry(tmpdir, branch)

            # 9. Capture commit SHA
            sha_result = self._run_git(
                ["rev-parse", "HEAD"], cwd=tmpdir
            )
            commit_sha = sha_result.stdout.strip()
            logger.info("✅ Push succeeded — commit %s", commit_sha[:8])

            return GitPushResult(
                success=True,
                commit_sha=commit_sha,
                remote_url=safe_url,
            )

        except RuntimeError as exc:
            # Auth / known operational errors
            logger.error("❌ Git operation failed: %s", exc)
            return GitPushResult(
                success=False,
                error_message=str(exc),
                remote_url=safe_url,
            )
        except subprocess.CalledProcessError as exc:
            logger.error(
                "❌ Git subprocess failed (rc=%d): %s",
                exc.returncode,
                self._redact_pat(exc.stderr or ""),
            )
            return GitPushResult(
                success=False,
                error_message=self._redact_pat(
                    f"git exited {exc.returncode}: {exc.stderr or 'unknown error'}"
                ),
                remote_url=safe_url,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("❌ Unexpected error during git push")
            return GitPushResult(
                success=False,
                error_message=f"Unexpected: {exc}",
                remote_url=safe_url,
            )
        finally:
            if tmpdir and os.path.isdir(tmpdir):
                shutil.rmtree(tmpdir, ignore_errors=True)
                logger.debug("🧹 Cleaned up temp dir %s", tmpdir)
