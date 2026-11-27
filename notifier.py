"""Private Telegram, macOS banner, and local-file alert delivery."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

import requests
from dotenv import load_dotenv


load_dotenv()

TELEGRAM_API_ROOT = "https://api.telegram.org/bot"
EDUCATIONAL_DISCLAIMER = (
    "Historical macro correlations are for informational research only, "
    "not financial advice."
)


class NotificationError(RuntimeError):
    """Raised when an alert cannot be delivered without exposing secrets."""


def format_inflection_message(events: Iterable[str]) -> str:
    """Format high-impact state changes as a private Markdown alert."""
    lines = ["*MACRO INFLECTION ALERT*", ""]
    lines.extend(f"• {event}" for event in events)
    lines.extend(("", EDUCATIONAL_DISCLAIMER))
    return "\n".join(lines)


def format_test_message() -> str:
    """Create a sample alert for verifying private Telegram delivery."""
    return (
        "*MACRO ALERT TEST*\n\n"
        "Private Telegram delivery is configured correctly.\n\n"
        f"{EDUCATIONAL_DISCLAIMER}"
    )


def format_summary_message(dashboard: str) -> str:
    """Add a Telegram title while preserving the complete dashboard snapshot."""
    message = f"*MACRO DASHBOARD SNAPSHOT*\n\n{dashboard}"
    if EDUCATIONAL_DISCLAIMER not in message:
        message = f"{message}\n\n{EDUCATIONAL_DISCLAIMER}"
    return message


class TelegramNotifier:
    """Send messages only to the configured private chat and record alerts."""

    def __init__(
        self,
        token: Optional[str] = None,
        chat_id: Optional[str] = None,
        log_path: Path = Path("data/alerts.log"),
    ) -> None:
        self.token = (token or os.getenv("TELEGRAM_BOT_TOKEN", "")).strip()
        self.chat_id = (chat_id or os.getenv("TELEGRAM_CHAT_ID", "")).strip()
        self.log_path = log_path

    def _log_alert(self, message: str) -> None:
        """Append a JSON-lines audit record without storing the bot token."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "chat_id": self.chat_id,
            "message": message,
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _macos_banner(message: str) -> None:
        """Display a local banner and safely no-op off macOS."""
        if sys.platform != "darwin":
            return
        summary = message.splitlines()[0].replace("*", "")[:180]
        summary = "Macro inflection alert" if not summary else summary
        escaped = summary.replace("\\", "\\\\").replace('"', '\\"')
        try:
            subprocess.run(
                [
                    "osascript",
                    "-e",
                    f'display notification "{escaped}" title "Macro Regime Alert"',
                ],
                check=False,
                capture_output=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            pass

    @staticmethod
    def _chunks(message: str, limit: int = 3900) -> List[str]:
        """Split long snapshots at line boundaries for Telegram's size limit."""
        chunks: List[str] = []
        remaining = message
        while len(remaining) > limit:
            split_at = remaining.rfind("\n", 0, limit)
            if split_at <= 0:
                split_at = limit
            chunks.append(remaining[:split_at])
            remaining = remaining[split_at:].lstrip("\n")
        if remaining:
            chunks.append(remaining)
        return chunks

    def _post(self, text: str) -> None:
        """Post one Markdown message with timeout and sanitized failures."""
        if not self.token:
            raise NotificationError("TELEGRAM_BOT_TOKEN is not configured")
        if not self.chat_id:
            raise NotificationError("TELEGRAM_CHAT_ID is not configured")
        url = f"{TELEGRAM_API_ROOT}{self.token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }
        try:
            response = requests.post(url, json=payload, timeout=10)
        except requests.RequestException as exc:
            raise NotificationError("Telegram request failed: network error") from exc
        if response.status_code != 200:
            raise NotificationError(
                f"Telegram delivery failed with HTTP {response.status_code}"
            )
        try:
            result = response.json()
        except ValueError as exc:
            raise NotificationError("Telegram returned an invalid response") from exc
        if not result.get("ok", False):
            raise NotificationError("Telegram rejected the alert payload")

    def send(self, message: str) -> int:
        """Log and display an alert, then deliver all chunks privately."""
        self._log_alert(message)
        self._macos_banner(message)
        chunks = self._chunks(message)
        for chunk in chunks:
            self._post(chunk)
        return len(chunks)


__all__ = [
    "EDUCATIONAL_DISCLAIMER",
    "NotificationError",
    "TelegramNotifier",
    "format_inflection_message",
    "format_summary_message",
    "format_test_message",
]
