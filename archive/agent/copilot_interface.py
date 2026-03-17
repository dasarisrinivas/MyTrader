"""Copilot Interface — lean subprocess wrapper for GitHub Copilot CLI.

Sends a compact prompt via `gh copilot -- -p "..."`.
Single round by default — the prompt is short (~2K chars) because our
code already parsed the logs. Copilot can read the actual files on disk
if it needs more detail (paths are included in the prompt).

No multi-round iteration — that caused the CLI to hang on long sessions.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

CST = ZoneInfo("America/Chicago")
AGENT_DIR = Path(__file__).resolve().parent
LOGS_DIR = AGENT_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("agent.copilot_interface")


@dataclass
class CopilotSession:
    """Result of a single Copilot diagnostic call."""
    session_id: str
    anomaly_id: str
    anomaly_type: str
    started_at: str
    prompt: str = ""
    response: str = ""
    duration_seconds: float = 0.0
    success: bool = False
    error: Optional[str] = None
    final_summary: str = ""
    suggestions: List[str] = field(default_factory=list)
    completed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "anomaly_id": self.anomaly_id,
            "anomaly_type": self.anomaly_type,
            "started_at": self.started_at,
            "prompt_chars": len(self.prompt),
            "response": self.response,
            "duration_seconds": self.duration_seconds,
            "success": self.success,
            "error": self.error,
            "final_summary": self.final_summary,
            "suggestions": self.suggestions,
            "completed": self.completed,
        }


class CopilotInterface:
    """Sends a single diagnostic prompt to `gh copilot` and captures the response."""

    def __init__(self, config: Dict[str, Any]):
        cop_cfg = config.get("copilot", {})
        self.timeout_seconds = cop_cfg.get("timeout_seconds", 90)
        self.retry_attempts = cop_cfg.get("retry_attempts", 2)
        self.retry_delay = cop_cfg.get("retry_delay_seconds", 5)

    # ─── Core CLI call ────────────────────────────────────────────────

    def _invoke_copilot(self, prompt: str) -> tuple[str, bool, Optional[str]]:
        """Send prompt to `gh copilot -- -p` and return (response, ok, error).

        Writes prompt to a temp file to avoid shell escaping issues,
        then reads it via $(cat ...).  Force-kills on timeout.
        """
        tmp_path = None
        proc = None

        for attempt in range(1, self.retry_attempts + 1):
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".txt", delete=False, prefix="copilot_"
                ) as tmp:
                    tmp.write(prompt)
                    tmp_path = tmp.name

                cmd = f'gh copilot -- -p "$(cat {tmp_path})"'
                log.info("Calling gh copilot (%d chars, attempt %d/%d, timeout %ds)...",
                         len(prompt), attempt, self.retry_attempts, self.timeout_seconds)

                proc = subprocess.Popen(
                    cmd, shell=True,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True,
                    env={**os.environ, "GH_PROMPT": "disable"},
                )

                try:
                    stdout, stderr = proc.communicate(timeout=self.timeout_seconds)
                except subprocess.TimeoutExpired:
                    log.warning("Copilot timed out after %ds — killing", self.timeout_seconds)
                    proc.kill()
                    try:
                        proc.communicate(timeout=5)
                    except Exception:
                        pass
                    if attempt < self.retry_attempts:
                        time.sleep(self.retry_delay)
                        continue
                    return "", False, f"Timeout after {self.timeout_seconds}s"

                if proc.returncode == 0 and stdout.strip():
                    cleaned = self._strip_stats(stdout)
                    if cleaned:
                        log.info("Copilot responded (%d chars)", len(cleaned))
                        return cleaned, True, None

                err = stderr.strip() or stdout.strip() or f"rc={proc.returncode}"
                if attempt < self.retry_attempts:
                    log.warning("Attempt %d failed: %s", attempt, err[:200])
                    time.sleep(self.retry_delay)
                else:
                    return "", False, f"Failed after {self.retry_attempts} attempts: {err[:300]}"

            except FileNotFoundError:
                return "", False, "gh CLI not found — brew install gh"
            except Exception as exc:
                return "", False, f"Error: {exc}"
            finally:
                if tmp_path:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass
                    tmp_path = None
                if proc and proc.poll() is None:
                    proc.kill()

        return "", False, "Exhausted retries"

    @staticmethod
    def _strip_stats(raw: str) -> str:
        """Strip trailing Copilot usage stats block."""
        lines = raw.split("\n")
        for i, line in enumerate(lines):
            s = line.strip()
            if s.startswith(("Total usage est:", "API time spent:",
                             "Total session time:", "Breakdown by AI model:",
                             "Total code changes:")):
                return "\n".join(lines[:i]).strip()
        return raw.strip()

    # ─── Public API ───────────────────────────────────────────────────

    def run_diagnostic_session(
        self,
        initial_prompt: str,
        anomaly_id: str,
        anomaly_type: str,
    ) -> CopilotSession:
        """Send one diagnostic prompt to Copilot and return the session."""
        session_id = f"{anomaly_type}_{anomaly_id}_{datetime.now(CST).strftime('%Y%m%d_%H%M%S')}"
        session = CopilotSession(
            session_id=session_id,
            anomaly_id=anomaly_id,
            anomaly_type=anomaly_type,
            started_at=datetime.now(CST).isoformat(),
            prompt=initial_prompt,
        )

        start = time.time()
        response, success, error = self._invoke_copilot(initial_prompt)
        session.duration_seconds = round(time.time() - start, 2)
        session.response = response
        session.success = success
        session.error = error
        session.completed = True

        if success:
            session.suggestions = self._extract_suggestions(response)
            session.final_summary = response[-1500:] if len(response) > 1500 else response
        else:
            session.final_summary = f"Copilot failed: {error}"

        self._save_transcript(session)
        return session

    def _extract_suggestions(self, text: str) -> List[str]:
        """Pull actionable items from Copilot's response."""
        suggestions = []
        for line in text.split("\n"):
            s = line.strip()
            if any(s.startswith(p) for p in [
                "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.",
                "- ", "* ", "• ", "→ ", "=> ",
            ]):
                clean = s.lstrip("0123456789.-*•→=> ").strip()
                if len(clean) > 10 and clean not in suggestions:
                    suggestions.append(clean)
        return suggestions[:15]

    def _save_transcript(self, session: CopilotSession) -> None:
        try:
            path = LOGS_DIR / f"copilot_{session.session_id}.json"
            with open(path, "w") as f:
                json.dump(session.to_dict(), f, indent=2, default=str)
            log.info("Transcript saved: %s", path.name)
        except Exception as exc:
            log.error("Failed to save transcript: %s", exc)

    def is_available(self) -> bool:
        try:
            r = subprocess.run(["gh", "--version"], capture_output=True, text=True, timeout=10)
            return r.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
