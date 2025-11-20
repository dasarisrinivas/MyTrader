"""Alerting system for critical trading events."""
from __future__ import annotations

import os
import smtplib
import requests
from email.mime.text import MIMEText
from typing import Optional
from ..utils.logger import logger

class Alerter:
    """Handles dispatching of alerts to various channels."""
    
    def __init__(self):
        self.slack_webhook = os.environ.get("SLACK_WEBHOOK_URL")
        self.email_host = os.environ.get("EMAIL_HOST")
        self.email_port = int(os.environ.get("EMAIL_PORT", "587"))
        self.email_user = os.environ.get("EMAIL_USER")
        self.email_password = os.environ.get("EMAIL_PASSWORD")
        self.email_to = os.environ.get("EMAIL_TO")
        
    def alert(self, title: str, message: str, level: str = "info") -> None:
        """Send an alert via configured channels."""
        
        # Always log
        log_msg = f"ALERT [{level.upper()}]: {title} - {message}"
        if level == "error" or level == "critical":
            logger.error(log_msg)
        elif level == "warning":
            logger.warning(log_msg)
        else:
            logger.info(log_msg)
            
        # Dispatch
        self._send_slack(title, message, level)
        self._send_email(title, message, level)
        
    def _send_slack(self, title: str, message: str, level: str) -> None:
        if not self.slack_webhook:
            return
            
        color = "#36a64f" # green
        if level == "warning":
            color = "#ffcc00"
        elif level == "error" or level == "critical":
            color = "#ff0000"
            
        payload = {
            "attachments": [
                {
                    "color": color,
                    "title": title,
                    "text": message,
                    "footer": "MyTrader Bot"
                }
            ]
        }
        
        try:
            response = requests.post(self.slack_webhook, json=payload, timeout=5)
            if response.status_code != 200:
                logger.error(f"Failed to send Slack alert: {response.text}")
        except Exception as e:
            logger.error(f"Error sending Slack alert: {e}")

    def _send_email(self, title: str, message: str, level: str) -> None:
        if not (self.email_host and self.email_user and self.email_password and self.email_to):
            return
            
        msg = MIMEText(message)
        msg['Subject'] = f"[{level.upper()}] MyTrader: {title}"
        msg['From'] = self.email_user
        msg['To'] = self.email_to
        
        try:
            with smtplib.SMTP(self.email_host, self.email_port) as server:
                server.starttls()
                server.login(self.email_user, self.email_password)
                server.send_message(msg)
        except Exception as e:
            logger.error(f"Error sending email alert: {e}")
