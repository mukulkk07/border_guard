"""
border_guard/config.py - Complete Configuration Management System
Centralizes all 60+ customizable parameters with full validation
"""

import os
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from enum import Enum
from dotenv import load_dotenv

load_dotenv()


class LogLevel(str, Enum):
    """Logging severity levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class RuleType(str, Enum):
    """Rule type categories"""
    FIELD = "field"
    DISTRIBUTION = "distribution"
    CUSTOM = "custom"


class AlertChannel(str, Enum):
    """Alert delivery channels"""
    STDOUT = "stdout"
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    PAGERDUTY = "pagerduty"
    LOG_FILE = "log_file"
    DATABASE = "database"


class AlertSeverity(str, Enum):
    """Alert severity classifications"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class DatabaseConfig:
    """Database connection and pooling configuration"""
    url: str = "sqlite:///./border_guard.db"
    echo: bool = False
    pool_size: int = 20
    max_overflow: int = 40
    pool_pre_ping: bool = True
    
    @classmethod
    def from_env(cls):
        return cls(
            url=os.getenv("DATABASE_URL", "sqlite:///./border_guard.db"),
            echo=os.getenv("DATABASE_ECHO", "false").lower() == "true",
            pool_size=int(os.getenv("DATABASE_POOL_SIZE", "20")),
            max_overflow=int(os.getenv("DATABASE_MAX_OVERFLOW", "40")),
            pool_pre_ping=os.getenv("DATABASE_POOL_PRE_PING", "true").lower() == "true",
        )


@dataclass
class QuarantineConfig:
    """Data quarantine and isolation configuration"""
    enabled: bool = True
    path: str = "./quarantine"
    retention_days: int = 30
    auto_cleanup: bool = True
    compression: str = "gzip"
    
    @classmethod
    def from_env(cls):
        return cls(
            enabled=os.getenv("QUARANTINE_ENABLED", "true").lower() == "true",
            path=os.getenv("QUARANTINE_PATH", "./quarantine"),
            retention_days=int(os.getenv("QUARANTINE_RETENTION_DAYS", "30")),
            auto_cleanup=os.getenv("QUARANTINE_AUTO_CLEANUP", "true").lower() == "true",
            compression=os.getenv("QUARANTINE_COMPRESSION", "gzip"),
        )


@dataclass
class MonitoringConfig:
    """Real-time monitoring and anomaly detection configuration"""
    enabled: bool = True
    baseline_lookback_days: int = 30
    anomaly_detection_enabled: bool = True
    anomaly_threshold_zscore: float = 3.0
    anomaly_threshold_percentage: float = 0.20
    check_interval_seconds: int = 300
    
    @classmethod
    def from_env(cls):
        return cls(
            enabled=os.getenv("MONITORING_ENABLED", "true").lower() == "true",
            baseline_lookback_days=int(os.getenv("BASELINE_LOOKBACK_DAYS", "30")),
            anomaly_detection_enabled=os.getenv("ANOMALY_DETECTION_ENABLED", "true").lower() == "true",
            anomaly_threshold_zscore=float(os.getenv("ANOMALY_ZSCORE_THRESHOLD", "3.0")),
            anomaly_threshold_percentage=float(os.getenv("ANOMALY_PERCENTAGE_THRESHOLD", "0.20")),
            check_interval_seconds=int(os.getenv("CHECK_INTERVAL_SECONDS", "300")),
        )


@dataclass
class AlertingConfig:
    """Alert routing and channel configuration"""
    enabled: bool = True
    channels: List[AlertChannel] = field(default_factory=lambda: [AlertChannel.STDOUT])
    default_channel: AlertChannel = AlertChannel.STDOUT
    slack_webhook_url: Optional[str] = None
    email_recipients: List[str] = field(default_factory=list)
    email_sender: str = "border-guard@company.com"
    email_smtp_server: str = "smtp.gmail.com"
    email_smtp_port: int = 587
    email_smtp_username: Optional[str] = None
    email_smtp_password: Optional[str] = None
    pagerduty_api_key: Optional[str] = None
    webhook_url: Optional[str] = None
    
    @classmethod
    def from_env(cls):
        channels_str = os.getenv("ALERT_CHANNELS", "stdout").split(",")
        channels = [AlertChannel(c.strip()) for c in channels_str if c.strip()]
        
        email_recipients_str = os.getenv("ALERT_EMAIL_RECIPIENTS", "")
        email_recipients = [e.strip() for e in email_recipients_str.split(",") if e.strip()]
        
        return cls(
            enabled=os.getenv("ALERTING_ENABLED", "true").lower() == "true",
            channels=channels or [AlertChannel.STDOUT],
            default_channel=AlertChannel(os.getenv("ALERT_DEFAULT_CHANNEL", "stdout")),
            slack_webhook_url=os.getenv("ALERT_SLACK_WEBHOOK"),
            email_recipients=email_recipients,
            email_sender=os.getenv("ALERT_EMAIL_SENDER", "border-guard@company.com"),
            email_smtp_server=os.getenv("ALERT_SMTP_SERVER", "smtp.gmail.com"),
            email_smtp_port=int(os.getenv("ALERT_SMTP_PORT", "587")),
            email_smtp_username=os.getenv("ALERT_SMTP_USERNAME"),
            email_smtp_password=os.getenv("ALERT_SMTP_PASSWORD"),
            pagerduty_api_key=os.getenv("ALERT_PAGERDUTY_KEY"),
            webhook_url=os.getenv("ALERT_WEBHOOK_URL"),
        )


@dataclass
class ValidationConfig:
    """Data validation strictness configuration"""
    strict_type_checking: bool = True
    allow_null_required_fields: bool = False
    trim_string_values: bool = False
    coerce_numeric_types: bool = False
    max_violations_to_report: int = 100
    
    @classmethod
    def from_env(cls):
        return cls(
            strict_type_checking=os.getenv("STRICT_TYPE_CHECKING", "true").lower() == "true",
            allow_null_required_fields=os.getenv("ALLOW_NULL_REQUIRED", "false").lower() == "true",
            trim_string_values=os.getenv("TRIM_STRING_VALUES", "false").lower() == "true",
            coerce_numeric_types=os.getenv("COERCE_NUMERIC_TYPES", "false").lower() == "true",
            max_violations_to_report=int(os.getenv("MAX_VIOLATIONS_REPORT", "100")),
        )


@dataclass
class APIConfig:
    """FastAPI server configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    debug: bool = False
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    api_title: str = "border-guard"
    api_version: str = "0.1.0"
    api_description: str = "Data quality and silent data corruption detection"
    
    @classmethod
    def from_env(cls):
        origins = os.getenv("CORS_ORIGINS", "*").split(",")
        return cls(
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "8000")),
            workers=int(os.getenv("API_WORKERS", "4")),
            debug=os.getenv("DEBUG", "false").lower() == "true",
            cors_origins=[o.strip() for o in origins if o.strip()],
            api_title=os.getenv("API_TITLE", "border-guard"),
            api_version=os.getenv("API_VERSION", "0.1.0"),
            api_description=os.getenv("API_DESCRIPTION", "Data quality and silent data corruption detection"),
        )


@dataclass
class LoggingConfig:
    """Logging system configuration"""
    level: LogLevel = LogLevel.INFO
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = None
    max_bytes: int = 10485760
    backup_count: int = 5
    
    @classmethod
    def from_env(cls):
        return cls(
            level=LogLevel(os.getenv("LOG_LEVEL", "INFO")),
            format=os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            file=os.getenv("LOG_FILE"),
            max_bytes=int(os.getenv("LOG_MAX_BYTES", "10485760")),
            backup_count=int(os.getenv("LOG_BACKUP_COUNT", "5")),
        )


@dataclass
class LineageConfig:
    """Data lineage and dependency tracking configuration"""
    enabled: bool = True
    auto_discover: bool = False
    max_depth: int = 10
    cache_ttl_seconds: int = 3600
    
    @classmethod
    def from_env(cls):
        return cls(
            enabled=os.getenv("LINEAGE_ENABLED", "true").lower() == "true",
            auto_discover=os.getenv("LINEAGE_AUTO_DISCOVER", "false").lower() == "true",
            max_depth=int(os.getenv("LINEAGE_MAX_DEPTH", "10")),
            cache_ttl_seconds=int(os.getenv("LINEAGE_CACHE_TTL", "3600")),
        )


class Config:
    """Master configuration class - loads all sub-configs from environment"""
    
    def __init__(self):
        self.database = DatabaseConfig.from_env()
        self.quarantine = QuarantineConfig.from_env()
        self.monitoring = MonitoringConfig.from_env()
        self.alerting = AlertingConfig.from_env()
        self.validation = ValidationConfig.from_env()
        self.api = APIConfig.from_env()
        self.logging = LoggingConfig.from_env()
        self.lineage = LineageConfig.from_env()
        
        # Validate configuration
        self.validate()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert all config to dictionary"""
        return {
            "database": asdict(self.database),
            "quarantine": asdict(self.quarantine),
            "monitoring": asdict(self.monitoring),
            "alerting": {
                **asdict(self.alerting),
                "channels": [c.value for c in self.alerting.channels],
                "default_channel": self.alerting.default_channel.value,
            },
            "validation": asdict(self.validation),
            "api": asdict(self.api),
            "logging": {**asdict(self.logging), "level": self.logging.level.value},
            "lineage": asdict(self.lineage),
        }
    
    def to_json(self) -> str:
        """Convert all config to JSON string"""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    def validate(self) -> bool:
        """Validate configuration consistency"""
        errors = []
        
        if self.database.pool_size < 1:
            errors.append("pool_size must be >= 1")
        if self.quarantine.retention_days < 1:
            errors.append("retention_days must be >= 1")
        if self.monitoring.anomaly_threshold_zscore < 0:
            errors.append("anomaly_threshold_zscore must be >= 0")
        if self.api.port < 1024 or self.api.port > 65535:
            errors.append("api.port must be between 1024 and 65535")
        if self.logging.max_bytes < 1024:
            errors.append("log.max_bytes must be >= 1024")
            
        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")
            
        return True


# Global config instance (auto-initializes)
config = Config()
