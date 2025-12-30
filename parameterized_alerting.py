"""border_guard/monitoring/parameterized_alerting.py - Flexible alerting system"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertChannel(str, Enum):
    """Alert channels"""
    STDOUT = "stdout"
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    PAGERDUTY = "pagerduty"
    LOG_FILE = "log_file"
    DATABASE = "database"


@dataclass
class Alert:
    """Alert object"""
    id: str
    dataset_name: str
    rule_name: str
    severity: AlertSeverity
    title: str
    message: str
    violations: List[Dict] = field(default_factory=list)
    affected_consumers: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    sent_to_channels: List[AlertChannel] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "dataset": self.dataset_name,
            "rule": self.rule_name,
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "violations": self.violations,
            "affected_consumers": self.affected_consumers,
            "created_at": self.created_at.isoformat(),
            "sent_channels": [c.value for c in self.sent_to_channels],
        }


class AlertNotifier(ABC):
    """Base class for alert channels"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
    
    @abstractmethod
    def send(self, alert: Alert) -> bool:
        """Send alert through channel"""
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """Test channel connectivity"""
        pass


class StdoutNotifier(AlertNotifier):
    """Print alerts to stdout"""
    
    def send(self, alert: Alert) -> bool:
        if not self.enabled:
            return False
        
        severity_color = {
            AlertSeverity.INFO: "\033[94m",      # Blue
            AlertSeverity.WARNING: "\033[93m",   # Yellow
            AlertSeverity.CRITICAL: "\033[91m",  # Red
            AlertSeverity.EMERGENCY: "\033[35m", # Magenta
        }
        
        reset_color = "\033[0m"
        color = severity_color.get(alert.severity, "")
        
        print(f"""
{color}
╔════════════════════════════════════════════════════════════╗
║ 🚨 ALERT: {alert.severity.value.upper()}
║ {alert.title}
╠════════════════════════════════════════════════════════════╣
║ Dataset:    {alert.dataset_name}
║ Rule:       {alert.rule_name}
║ Severity:   {alert.severity.value}
║ Time:       {alert.created_at}
║─────────────────────────────────────────────────────────────
║ Message:
║ {alert.message}
║─────────────────────────────────────────────────────────────
║ Violations: {len(alert.violations)}
║ Consumers:  {', '.join(alert.affected_consumers) or 'None'}
╚════════════════════════════════════════════════════════════╝
{reset_color}
        """)
        
        return True
    
    def test_connection(self) -> bool:
        print("✅ Stdout notifier ready")
        return True


class EmailNotifier(AlertNotifier):
    """Send alerts via email"""
    
    def send(self, alert: Alert) -> bool:
        if not self.enabled:
            return False
        
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            recipients = self.config.get("recipients", [])
            if not recipients:
                logger.warning("No email recipients configured")
                return False
            
            # Compose email
            msg = MIMEMultipart()
            msg["From"] = self.config.get("sender", "border-guard@company.com")
            msg["To"] = ", ".join(recipients)
            msg["Subject"] = f"[border-guard] {alert.severity.value.upper()}: {alert.title}"
            
            # HTML body
            html = f"""
            <html>
                <body style="font-family: Arial, sans-serif;">
                    <h2 style="color: {'red' if alert.severity == AlertSeverity.CRITICAL else 'orange'};">
                        🚨 {alert.severity.value.upper()}: {alert.title}
                    </h2>
                    <table style="border: 1px solid #ddd; border-collapse: collapse;">
                        <tr>
                            <td style="padding: 10px; border: 1px solid #ddd;"><b>Dataset:</b></td>
                            <td style="padding: 10px; border: 1px solid #ddd;">{alert.dataset_name}</td>
                        </tr>
                        <tr>
                            <td style="padding: 10px; border: 1px solid #ddd;"><b>Rule:</b></td>
                            <td style="padding: 10px; border: 1px solid #ddd;">{alert.rule_name}</td>
                        </tr>
                        <tr>
                            <td style="padding: 10px; border: 1px solid #ddd;"><b>Time:</b></td>
                            <td style="padding: 10px; border: 1px solid #ddd;">{alert.created_at}</td>
                        </tr>
                        <tr>
                            <td style="padding: 10px; border: 1px solid #ddd;"><b>Violations:</b></td>
                            <td style="padding: 10px; border: 1px solid #ddd;">{len(alert.violations)}</td>
                        </tr>
                        <tr>
                            <td style="padding: 10px; border: 1px solid #ddd;"><b>Affected:</b></td>
                            <td style="padding: 10px; border: 1px solid #ddd;">
                                {', '.join(alert.affected_consumers) or 'None'}
                            </td>
                        </tr>
                    </table>
                    <h3>Message:</h3>
                    <p>{alert.message}</p>
                </body>
            </html>
            """
            
            msg.attach(MIMEText(html, "html"))
            
            # Send email
            server = smtplib.SMTP(
                self.config.get("smtp_server", "smtp.gmail.com"),
                self.config.get("smtp_port", 587)
            )
            server.starttls()
            server.login(
                self.config.get("smtp_username"),
                self.config.get("smtp_password")
            )
            server.sendmail(msg["From"], recipients, msg.as_string())
            server.quit()
            
            logger.info(f"Email alert sent to {recipients}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False
    
    def test_connection(self) -> bool:
        try:
            import smtplib
            server = smtplib.SMTP(
                self.config.get("smtp_server", "smtp.gmail.com"),
                self.config.get("smtp_port", 587)
            )
            server.starttls()
            server.quit()
            return True
        except Exception as e:
            logger.error(f"Email connection failed: {e}")
            return False


class SlackNotifier(AlertNotifier):
    """Send alerts to Slack"""
    
    def send(self, alert: Alert) -> bool:
        if not self.enabled:
            return False
        
        try:
            import requests
            
            webhook_url = self.config.get("webhook_url")
            if not webhook_url:
                logger.warning("Slack webhook URL not configured")
                return False
            
            # Determine color based on severity
            color_map = {
                AlertSeverity.INFO: "#0099FF",
                AlertSeverity.WARNING: "#FFAA00",
                AlertSeverity.CRITICAL: "#FF0000",
                AlertSeverity.EMERGENCY: "#AA00FF",
            }
            
            payload = {
                "attachments": [
                    {
                        "color": color_map.get(alert.severity, "#808080"),
                        "title": alert.title,
                        "text": alert.message,
                        "fields": [
                            {"title": "Dataset", "value": alert.dataset_name, "short": True},
                            {"title": "Rule", "value": alert.rule_name, "short": True},
                            {"title": "Severity", "value": alert.severity.value, "short": True},
                            {"title": "Violations", "value": str(len(alert.violations)), "short": True},
                            {
                                "title": "Affected Consumers",
                                "value": ", ".join(alert.affected_consumers) or "None",
                                "short": False
                            },
                        ],
                        "ts": int(alert.created_at.timestamp()),
                    }
                ]
            }
            
            response = requests.post(webhook_url, json=payload)
            success = response.status_code == 200
            
            if success:
                logger.info("Slack alert sent successfully")
            else:
                logger.error(f"Slack alert failed: {response.text}")
            
            return success
        
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False
    
    def test_connection(self) -> bool:
        try:
            import requests
            webhook_url = self.config.get("webhook_url")
            if not webhook_url:
                return False
            
            test_payload = {"text": "✅ border-guard Slack connection test"}
            response = requests.post(webhook_url, json=test_payload)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Slack connection test failed: {e}")
            return False


class WebhookNotifier(AlertNotifier):
    """Send alerts to custom webhook"""
    
    def send(self, alert: Alert) -> bool:
        if not self.enabled:
            return False
        
        try:
            import requests
            
            webhook_url = self.config.get("url")
            if not webhook_url:
                logger.warning("Webhook URL not configured")
                return False
            
            headers = self.config.get("headers", {"Content-Type": "application/json"})
            
            response = requests.post(
                webhook_url,
                json=alert.to_dict(),
                headers=headers,
                timeout=10
            )
            
            success = response.status_code in [200, 201, 202]
            
            if success:
                logger.info(f"Webhook alert sent to {webhook_url}")
            else:
                logger.error(f"Webhook failed: {response.text}")
            
            return success
        
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False
    
    def test_connection(self) -> bool:
        try:
            import requests
            webhook_url = self.config.get("url")
            if not webhook_url:
                return False
            
            response = requests.get(webhook_url + "?test=true", timeout=5)
            return response.status_code in [200, 201, 404]
        except Exception as e:
            logger.error(f"Webhook connection test failed: {e}")
            return False


@dataclass
class ParameterizedAlertingSystem:
    """Main alerting system with multiple channels"""
    
    notifiers: Dict[AlertChannel, AlertNotifier] = field(default_factory=dict)
    channel_config: Dict[AlertChannel, Dict[str, Any]] = field(default_factory=dict)
    severity_routing: Dict[AlertSeverity, List[AlertChannel]] = field(default_factory=dict)
    alert_history: List[Alert] = field(default_factory=list)
    max_history: int = 1000
    
    def register_notifier(self, channel: AlertChannel, notifier: AlertNotifier):
        """Register alert channel"""
        self.notifiers[channel] = notifier
        logger.info(f"Registered alert channel: {channel.value}")
    
    def set_severity_routing(self, severity: AlertSeverity, channels: List[AlertChannel]):
        """Configure which channels to use for each severity level"""
        self.severity_routing[severity] = channels
        logger.info(f"Set routing for {severity.value}: {[c.value for c in channels]}")
    
    def send_alert(self, alert: Alert, override_channels: Optional[List[AlertChannel]] = None) -> bool:
        """Send alert through configured channels"""
        
        channels = override_channels or self.severity_routing.get(
            alert.severity,
            list(self.notifiers.keys())
        )
        
        success = True
        for channel in channels:
            if channel not in self.notifiers:
                logger.warning(f"Channel {channel.value} not registered")
                continue
            
            notifier = self.notifiers[channel]
            try:
                if notifier.send(alert):
                    alert.sent_to_channels.append(channel)
                else:
                    success = False
            except Exception as e:
                logger.error(f"Error sending alert via {channel.value}: {e}")
                success = False
        
        # Store in history
        self.alert_history.append(alert)
        if len(self.alert_history) > self.max_history:
            self.alert_history.pop(0)
        
        return success
    
    def test_all_channels(self) -> Dict[AlertChannel, bool]:
        """Test connectivity for all channels"""
        results = {}
        for channel, notifier in self.notifiers.items():
            results[channel] = notifier.test_connection()
            status = "✅" if results[channel] else "❌"
            logger.info(f"{status} {channel.value} connection test")
        return results
    
    def get_alert_history(self, dataset: Optional[str] = None, 
                          severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get alert history with optional filtering"""
        results = self.alert_history
        
        if dataset:
            results = [a for a in results if a.dataset_name == dataset]
        
        if severity:
            results = [a for a in results if a.severity == severity]
        
        return results
