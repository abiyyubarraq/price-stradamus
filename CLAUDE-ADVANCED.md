# Price Stradamus - Advanced Guide (Phase 2-6)

---

## 📍 **ADVANCED TOPICS: Phases 2-6**

This guide contains **advanced topics** for scaling Price Stradamus from personal use to cloud deployment.

**Current Development**: For Phase 1 essentials, see [CLAUDE.md](CLAUDE.md).

### Phase Overview

- **Phase 1** (✅ Current): Personal use, great predictions → See [CLAUDE.md](CLAUDE.md)
- **Phase 2**: AutoML & Optimization (<1k users)
- **Phase 3**: Production Hardening (<10k users)
- **Phase 4**: API & Multi-User (<100k users)
- **Phase 5**: Real-Time Streaming (<500k users)
- **Phase 6**: Cloud Deployment (1M+ users)

**For daily development**, use [CLAUDE.md](CLAUDE.md). This file is for planning future phases.

---

## Table of Contents

1. [Security Guidelines (Advanced)](#security-guidelines-phase-1-basics--phase-3-4-advanced)
2. [Dependency Management](#dependency-management-phase-2)
3. [Code Ownership](#code-ownership-codeowners-phase-4)
4. [Breaking Change Policy](#breaking-change-policy-phase-3-4)
5. [Deprecation Policy](#deprecation-policy-phase-3-4)
6. [Performance Testing](#performance-testing-guidelines-phase-3)
7. [Structured Concurrency](#structured-concurrency-taskgroup-phase-2-3)
8. [CLI Accessibility](#cli-accessibility-guidelines-phase-4)

---

├────────────────────────────────────────────────────────────┤
│  Network Layer    │ TLS 1.3, Firewall, Rate Limiting       │
├───────────────────┼────────────────────────────────────────┤
│  Application      │ Input Validation, Output Encoding      │
├───────────────────┼────────────────────────────────────────┤
│  Authentication   │ API Keys, JWT, OAuth 2.0               │
├───────────────────┼────────────────────────────────────────┤
│  Data Layer       │ Encryption at Rest, Parameterized SQL  │
├───────────────────┼────────────────────────────────────────┤
│  Infrastructure   │ Secrets Management, Least Privilege    │
└───────────────────┴────────────────────────────────────────┘
```

### Secrets Management

**Never commit secrets to version control.** Use environment variables or secret managers.

```python
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from cryptography.fernet import Fernet
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class SecretProvider(Protocol):
    """Protocol for secret providers."""

    def get_secret(self, key: str) -> str:
        """Retrieve a secret by key."""
        ...


class EnvironmentSecretProvider:
    """Retrieve secrets from environment variables."""

    def get_secret(self, key: str) -> str:
        """Get secret from environment variable.

        Args:
            key: Environment variable name

        Returns:
            Secret value

        Raises:
            ValueError: If secret not found
        """
        value = os.environ.get(key)
        if value is None:
            raise ValueError(f"Secret {key} not found in environment")
        return value


class EncryptedFileSecretProvider:
    """Retrieve secrets from encrypted file."""

    def __init__(self, secrets_file: Path, key: bytes) -> None:
        """Initialize with encrypted secrets file.

        Args:
            secrets_file: Path to encrypted secrets file
            key: Fernet encryption key
        """
        self.secrets_file = secrets_file
        self.fernet = Fernet(key)
        self._cache: dict[str, str] = {}

    def get_secret(self, key: str) -> str:
        """Get secret from encrypted file.

        Args:
            key: Secret key name

        Returns:
            Decrypted secret value
        """
        if not self._cache:
            self._load_secrets()
        if key not in self._cache:
            raise ValueError(f"Secret {key} not found")
        return self._cache[key]

    def _load_secrets(self) -> None:
        """Load and decrypt secrets file."""
        import json

        encrypted_data = self.secrets_file.read_bytes()
        decrypted_data = self.fernet.decrypt(encrypted_data)
        self._cache = json.loads(decrypted_data.decode())


class SecureSettings(BaseSettings):
    """Application settings with secure secret handling."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database (use SecretStr to prevent logging)
    database_password: SecretStr
    database_url: SecretStr

    # API Keys
    binance_api_key: SecretStr
    binance_api_secret: SecretStr

    # Encryption
    encryption_key: SecretStr

    def get_database_url(self) -> str:
        """Get database URL with password revealed."""
        return self.database_url.get_secret_value()


@lru_cache(maxsize=1)
def get_settings() -> SecureSettings:
    """Get cached settings instance."""
    return SecureSettings()


# Usage
settings = get_settings()
# Password is masked in logs
print(settings.database_password)  # SecretStr('**********')
# Explicitly reveal when needed
password = settings.database_password.get_secret_value()
```

### OWASP Top 10 Mitigations

#### 1. Injection Prevention (A03:2021)

```python
from __future__ import annotations

from typing import Any

import asyncpg


class SecureDatabaseClient:
    """Database client with injection prevention."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self.pool = pool

    async def get_predictions(
        self,
        symbol: str,
        model_name: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get predictions with parameterized query.

        NEVER use string formatting for SQL queries.
        ALWAYS use parameterized queries.
        """
        # GOOD: Parameterized query
        query = """
            SELECT *
            FROM ml_data.predictions
            WHERE symbol = $1
              AND model_name = $2
            ORDER BY timestamp DESC
            LIMIT $3
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, symbol, model_name, limit)
            return [dict(row) for row in rows]

    # BAD - NEVER DO THIS
    # async def get_predictions_unsafe(self, symbol: str) -> list:
    #     query = f"SELECT * FROM predictions WHERE symbol = '{symbol}'"
    #     # SQL injection vulnerability!


class InputValidator:
    """Validate and sanitize user input."""

    # Allowlist of valid symbols
    VALID_SYMBOLS = frozenset({"BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"})

    # Allowlist of valid timeframes
    VALID_TIMEFRAMES = frozenset({"1m", "5m", "15m", "1h", "4h", "1d"})

    @classmethod
    def validate_symbol(cls, symbol: str) -> str:
        """Validate trading symbol.

        Args:
            symbol: Trading pair symbol

        Returns:
            Validated symbol (uppercase)

        Raises:
            ValueError: If symbol not in allowlist
        """
        symbol_upper = symbol.upper().strip()
        if symbol_upper not in cls.VALID_SYMBOLS:
            raise ValueError(
                f"Invalid symbol: {symbol}. "
                f"Must be one of: {cls.VALID_SYMBOLS}"
            )
        return symbol_upper

    @classmethod
    def validate_timeframe(cls, timeframe: str) -> str:
        """Validate timeframe."""
        tf_lower = timeframe.lower().strip()
        if tf_lower not in cls.VALID_TIMEFRAMES:
            raise ValueError(
                f"Invalid timeframe: {timeframe}. "
                f"Must be one of: {cls.VALID_TIMEFRAMES}"
            )
        return tf_lower

    @classmethod
    def validate_positive_int(
        cls,
        value: int,
        name: str,
        max_value: int = 10000,
    ) -> int:
        """Validate positive integer within bounds."""
        if not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive")
        if value > max_value:
            raise ValueError(f"{name} must not exceed {max_value}")
        return value
```

#### 2. Broken Authentication Prevention (A07:2021)

```python
from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError


@dataclass
class APIKey:
    """API key with metadata."""

    key_id: str
    key_hash: str
    created_at: datetime
    expires_at: datetime | None
    rate_limit: int
    scopes: list[str]


class SecureAuthManager:
    """Secure authentication manager."""

    def __init__(
        self,
        jwt_secret: str,
        jwt_algorithm: str = "HS256",
        token_expiry_hours: int = 24,
    ) -> None:
        self.jwt_secret = jwt_secret
        self.jwt_algorithm = jwt_algorithm
        self.token_expiry = timedelta(hours=token_expiry_hours)
        self.password_hasher = PasswordHasher(
            time_cost=3,
            memory_cost=65536,
            parallelism=4,
        )

    def generate_api_key(self, prefix: str = "ps") -> tuple[str, str]:
        """Generate a secure API key.

        Returns:
            Tuple of (full_key_to_give_user, key_hash_to_store)
        """
        # Generate 32 random bytes (256 bits)
        random_bytes = secrets.token_bytes(32)

        # Create key with prefix for identification
        key_id = secrets.token_hex(8)
        key_secret = secrets.token_urlsafe(32)
        full_key = f"{prefix}_{key_id}_{key_secret}"

        # Hash for storage (never store raw key)
        key_hash = hashlib.sha256(full_key.encode()).hexdigest()

        return full_key, key_hash

    def verify_api_key(self, provided_key: str, stored_hash: str) -> bool:
        """Verify API key using constant-time comparison.

        Args:
            provided_key: Key provided by client
            stored_hash: Hash stored in database

        Returns:
            True if key is valid
        """
        provided_hash = hashlib.sha256(provided_key.encode()).hexdigest()
        return hmac.compare_digest(provided_hash, stored_hash)

    def hash_password(self, password: str) -> str:
        """Hash password using Argon2id.

        Argon2id is the recommended algorithm for password hashing.
        It's resistant to side-channel and GPU attacks.
        """
        return self.password_hasher.hash(password)

    def verify_password(self, password: str, hash: str) -> bool:
        """Verify password against hash."""
        try:
            self.password_hasher.verify(hash, password)
            return True
        except VerifyMismatchError:
            return False

    def create_jwt_token(
        self,
        user_id: str,
        scopes: list[str],
        extra_claims: dict | None = None,
    ) -> str:
        """Create JWT token with security best practices.

        Args:
            user_id: User identifier
            scopes: Permission scopes
            extra_claims: Additional claims

        Returns:
            Signed JWT token
        """
        now = datetime.now(timezone.utc)
        payload = {
            "sub": user_id,
            "iat": now,
            "exp": now + self.token_expiry,
            "nbf": now,  # Not valid before
            "jti": secrets.token_hex(16),  # Unique token ID
            "scopes": scopes,
            **(extra_claims or {}),
        }
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)

    def verify_jwt_token(self, token: str) -> dict:
        """Verify and decode JWT token.

        Args:
            token: JWT token string

        Returns:
            Decoded payload

        Raises:
            jwt.InvalidTokenError: If token is invalid
        """
        return jwt.decode(
            token,
            self.jwt_secret,
            algorithms=[self.jwt_algorithm],
            options={
                "require": ["exp", "iat", "sub"],
                "verify_exp": True,
                "verify_iat": True,
                "verify_nbf": True,
            },
        )


class RateLimiter:
    """Token bucket rate limiter to prevent brute force attacks."""

    def __init__(
        self,
        rate: float,  # tokens per second
        capacity: int,  # maximum tokens
    ) -> None:
        self.rate = rate
        self.capacity = capacity
        self._buckets: dict[str, tuple[float, float]] = {}

    def is_allowed(self, key: str) -> bool:
        """Check if request is allowed.

        Args:
            key: Identifier (e.g., IP address, user ID)

        Returns:
            True if request is allowed
        """
        now = time.monotonic()

        if key not in self._buckets:
            self._buckets[key] = (self.capacity - 1, now)
            return True

        tokens, last_update = self._buckets[key]

        # Add tokens based on time elapsed
        elapsed = now - last_update
        tokens = min(self.capacity, tokens + elapsed * self.rate)

        if tokens >= 1:
            self._buckets[key] = (tokens - 1, now)
            return True

        return False


# Usage example
auth_manager = SecureAuthManager(jwt_secret="your-256-bit-secret")

# Generate API key for user
api_key, key_hash = auth_manager.generate_api_key()
print(f"Give to user: {api_key}")
print(f"Store in DB: {key_hash}")

# Create JWT
token = auth_manager.create_jwt_token(
    user_id="user123",
    scopes=["predictions:read", "models:list"],
)
```

#### 3. Sensitive Data Exposure Prevention (A02:2021)

```python
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SensitiveDataHandler:
    """Handle sensitive data with proper masking and hashing."""

    # Patterns for sensitive data detection
    SENSITIVE_PATTERNS: dict[str, re.Pattern] = field(default_factory=lambda: {
        "api_key": re.compile(r"(ps_[a-zA-Z0-9_-]{40,})", re.IGNORECASE),
        "password": re.compile(r"password[\"']?\s*[:=]\s*[\"']?([^\"'\s]+)", re.IGNORECASE),
        "secret": re.compile(r"secret[\"']?\s*[:=]\s*[\"']?([^\"'\s]+)", re.IGNORECASE),
        "token": re.compile(r"(eyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*)", re.IGNORECASE),
        "private_key": re.compile(r"-----BEGIN.*PRIVATE KEY-----", re.IGNORECASE),
    })

    def mask_value(self, value: str, visible_chars: int = 4) -> str:
        """Mask sensitive value, showing only first/last chars.

        Args:
            value: Sensitive string to mask
            visible_chars: Number of visible characters at start/end

        Returns:
            Masked string
        """
        if len(value) <= visible_chars * 2:
            return "*" * len(value)
        return f"{value[:visible_chars]}{'*' * (len(value) - visible_chars * 2)}{value[-visible_chars:]}"

    def sanitize_log_message(self, message: str) -> str:
        """Remove sensitive data from log messages.

        Args:
            message: Log message to sanitize

        Returns:
            Sanitized message
        """
        sanitized = message
        for name, pattern in self.SENSITIVE_PATTERNS.items():
            sanitized = pattern.sub(f"[REDACTED_{name.upper()}]", sanitized)
        return sanitized

    def hash_for_logging(self, value: str) -> str:
        """Create a hash of sensitive data for correlation in logs.

        Args:
            value: Sensitive value

        Returns:
            Short hash for correlation
        """
        return hashlib.sha256(value.encode()).hexdigest()[:12]

    def safe_dict_for_logging(self, data: dict[str, Any]) -> dict[str, Any]:
        """Create a safe copy of dict for logging.

        Args:
            data: Dictionary potentially containing sensitive data

        Returns:
            Sanitized dictionary
        """
        sensitive_keys = {
            "password", "secret", "token", "api_key", "apikey",
            "authorization", "auth", "credentials", "private_key",
        }
        result = {}
        for key, value in data.items():
            key_lower = key.lower()
            if any(s in key_lower for s in sensitive_keys):
                if isinstance(value, str):
                    result[key] = self.mask_value(value)
                else:
                    result[key] = "[REDACTED]"
            elif isinstance(value, dict):
                result[key] = self.safe_dict_for_logging(value)
            else:
                result[key] = value
        return result


class SecureLogger:
    """Logger that automatically redacts sensitive information."""

    def __init__(self) -> None:
        from loguru import logger
        self._logger = logger
        self._handler = SensitiveDataHandler()

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message with sensitive data redacted."""
        safe_message = self._handler.sanitize_log_message(message)
        safe_kwargs = self._handler.safe_dict_for_logging(kwargs)
        self._logger.info(safe_message, **safe_kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message with sensitive data redacted."""
        safe_message = self._handler.sanitize_log_message(message)
        safe_kwargs = self._handler.safe_dict_for_logging(kwargs)
        self._logger.error(safe_message, **safe_kwargs)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message with sensitive data redacted."""
        safe_message = self._handler.sanitize_log_message(message)
        safe_kwargs = self._handler.safe_dict_for_logging(kwargs)
        self._logger.debug(safe_message, **safe_kwargs)


# Usage
secure_logger = SecureLogger()
secure_logger.info(
    "User login attempt",
    user_id="user123",
    api_key="ps_abc123_very_secret_key_here",  # Will be masked
    password="supersecret",  # Will be masked
)
```

### File Security

```python
from __future__ import annotations

import os
import stat
from pathlib import Path


class SecureFileHandler:
    """Handle files with proper security permissions."""

    @staticmethod
    def create_secure_file(path: Path, content: str) -> None:
        """Create file with restricted permissions (600).

        Args:
            path: File path
            content: File content
        """
        # Create parent directories if needed
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write content
        path.write_text(content)

        # Set permissions to owner read/write only (600)
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)

    @staticmethod
    def create_secure_directory(path: Path) -> None:
        """Create directory with restricted permissions (700).

        Args:
            path: Directory path
        """
        path.mkdir(parents=True, exist_ok=True)
        os.chmod(path, stat.S_IRWXU)  # rwx------

    @staticmethod
    def validate_path(base_dir: Path, user_path: str) -> Path:
        """Validate path to prevent directory traversal.

        Args:
            base_dir: Base directory that should contain the path
            user_path: User-provided path

        Returns:
            Safe absolute path

        Raises:
            ValueError: If path escapes base directory
        """
        # Resolve to absolute path
        full_path = (base_dir / user_path).resolve()

        # Verify path is within base directory
        try:
            full_path.relative_to(base_dir.resolve())
        except ValueError:
            raise ValueError(
                f"Path '{user_path}' escapes base directory. "
                "Directory traversal not allowed."
            )

        return full_path


# Usage
handler = SecureFileHandler()

# Create secure config file
handler.create_secure_file(
    Path("config/secrets.json"),
    '{"api_key": "secret"}',
)

# Validate user-provided path
base = Path("/app/models")
try:
    safe_path = handler.validate_path(base, "../../../etc/passwd")
except ValueError as e:
    print(f"Blocked: {e}")
```

---

## Dependency Management [PHASE 2]

### Version Pinning Strategy

**Pin all dependencies** to exact versions in production for reproducibility.

```toml
# pyproject.toml
[project]
name = "price-stradamus"
version = "1.0.0"
requires-python = ">=3.13"

dependencies = [
    # Core dependencies - pin to exact versions
    "darts==0.30.0",
    "pandas==2.2.0",
    "numpy==1.26.4",
    "torch==2.2.0",
    "scikit-learn==1.4.0",

    # Database
    "asyncpg==0.29.0",
    "sqlalchemy==2.0.25",
    "alembic==1.13.1",

    # API & Web
    "aiohttp==3.9.3",
    "pydantic==2.6.0",
    "pydantic-settings==2.1.0",

    # Utilities
    "loguru==0.7.2",
    "rich==13.7.0",
    "typer==0.9.0",
]

[project.optional-dependencies]
dev = [
    # Testing
    "pytest==8.0.0",
    "pytest-asyncio==0.23.4",
    "pytest-cov==4.1.0",
    "pytest-mock==3.12.0",

    # Code quality
    "ruff==0.2.1",
    "pyright==1.1.350",
    "pre-commit==3.6.0",

    # Security scanning
    "bandit==1.7.7",
    "safety==2.3.5",
    "pip-audit==2.7.2",
]
```

### Dependency Audit Pipeline

```python
from __future__ import annotations

import subprocess
import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path


class VulnerabilitySeverity(str, Enum):
    """Vulnerability severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


@dataclass
class Vulnerability:
    """Security vulnerability information."""

    package: str
    installed_version: str
    affected_versions: str
    vulnerability_id: str
    severity: VulnerabilitySeverity
    description: str
    fix_version: str | None


@dataclass
class AuditResult:
    """Dependency audit result."""

    timestamp: datetime
    vulnerabilities: list[Vulnerability]
    packages_scanned: int

    @property
    def has_critical(self) -> bool:
        return any(v.severity == VulnerabilitySeverity.CRITICAL for v in self.vulnerabilities)

    @property
    def has_high(self) -> bool:
        return any(v.severity == VulnerabilitySeverity.HIGH for v in self.vulnerabilities)


class DependencyAuditor:
    """Audit dependencies for security vulnerabilities."""

    def __init__(self, project_dir: Path) -> None:
        self.project_dir = project_dir

    def run_pip_audit(self) -> AuditResult:
        """Run pip-audit for vulnerability scanning.

        Returns:
            Audit result with vulnerabilities
        """
        result = subprocess.run(
            ["pip-audit", "--format", "json", "--strict"],
            capture_output=True,
            text=True,
            cwd=self.project_dir,
        )

        vulnerabilities = []
        if result.stdout:
            data = json.loads(result.stdout)
            for vuln in data.get("vulnerabilities", []):
                vulnerabilities.append(
                    Vulnerability(
                        package=vuln["name"],
                        installed_version=vuln["version"],
                        affected_versions=vuln.get("affected_versions", ""),
                        vulnerability_id=vuln["id"],
                        severity=VulnerabilitySeverity(
                            vuln.get("severity", "unknown").lower()
                        ),
                        description=vuln.get("description", ""),
                        fix_version=vuln.get("fix_versions", [None])[0],
                    )
                )

        return AuditResult(
            timestamp=datetime.now(),
            vulnerabilities=vulnerabilities,
            packages_scanned=len(data.get("dependencies", [])),
        )

    def run_safety_check(self) -> AuditResult:
        """Run Safety CLI for vulnerability scanning."""
        result = subprocess.run(
            ["safety", "check", "--json"],
            capture_output=True,
            text=True,
            cwd=self.project_dir,
        )

        vulnerabilities = []
        if result.stdout:
            data = json.loads(result.stdout)
            for vuln in data:
                vulnerabilities.append(
                    Vulnerability(
                        package=vuln[0],
                        installed_version=vuln[2],
                        affected_versions=vuln[1],
                        vulnerability_id=vuln[4],
                        severity=VulnerabilitySeverity.UNKNOWN,
                        description=vuln[3],
                        fix_version=None,
                    )
                )

        return AuditResult(
            timestamp=datetime.now(),
            vulnerabilities=vulnerabilities,
            packages_scanned=0,  # Safety doesn't report this
        )

    def generate_report(self, result: AuditResult) -> str:
        """Generate human-readable vulnerability report.

        Args:
            result: Audit result

        Returns:
            Formatted report string
        """
        lines = [
            "# Dependency Security Audit Report",
            f"Date: {result.timestamp.isoformat()}",
            f"Packages Scanned: {result.packages_scanned}",
            f"Vulnerabilities Found: {len(result.vulnerabilities)}",
            "",
        ]

        if not result.vulnerabilities:
            lines.append("✅ No vulnerabilities found!")
            return "\n".join(lines)

        # Group by severity
        by_severity: dict[VulnerabilitySeverity, list[Vulnerability]] = {}
        for vuln in result.vulnerabilities:
            by_severity.setdefault(vuln.severity, []).append(vuln)

        for severity in [
            VulnerabilitySeverity.CRITICAL,
            VulnerabilitySeverity.HIGH,
            VulnerabilitySeverity.MEDIUM,
            VulnerabilitySeverity.LOW,
        ]:
            vulns = by_severity.get(severity, [])
            if vulns:
                lines.append(f"\n## {severity.value.upper()} ({len(vulns)})")
                for v in vulns:
                    lines.extend([
                        f"- **{v.package}** ({v.installed_version})",
                        f"  - ID: {v.vulnerability_id}",
                        f"  - Fix: {v.fix_version or 'No fix available'}",
                        f"  - {v.description[:100]}...",
                    ])

        return "\n".join(lines)


# CI/CD integration
def check_dependencies_ci() -> int:
    """Run dependency audit in CI/CD pipeline.

    Returns:
        Exit code (0 for success, 1 for vulnerabilities)
    """
    auditor = DependencyAuditor(Path("."))
    result = auditor.run_pip_audit()

    print(auditor.generate_report(result))

    if result.has_critical or result.has_high:
        print("\n❌ FAILED: Critical or High vulnerabilities found!")
        return 1

    print("\n✅ PASSED: No critical vulnerabilities")
    return 0
```

### Lock File Management

```bash
# Generate lock file for reproducible builds
uv pip compile pyproject.toml -o requirements.lock

# Install from lock file
uv pip sync requirements.lock

# Update all dependencies
uv pip compile pyproject.toml -o requirements.lock --upgrade

# Update specific package
uv pip compile pyproject.toml -o requirements.lock --upgrade-package pandas
```

---

## Code Ownership (CODEOWNERS) [PHASE 4]

Define code ownership for review requirements:

```
# .github/CODEOWNERS

# Default owner for everything
* @price-stradamus/core-team

# Data pipeline
src/price_stradamus/data/ @price-stradamus/data-team
tests/test_data/ @price-stradamus/data-team

# Models
src/price_stradamus/models/ @price-stradamus/ml-team
tests/test_models/ @price-stradamus/ml-team

# Evaluation
src/price_stradamus/evaluation/ @price-stradamus/ml-team @price-stradamus/quant-team

# Infrastructure
docker/ @price-stradamus/devops-team
.github/ @price-stradamus/devops-team
alembic/ @price-stradamus/data-team

# Documentation requires ML team review
*.md @price-stradamus/ml-team
context/ @price-stradamus/ml-team

# Configuration files
pyproject.toml @price-stradamus/core-team
*.yaml @price-stradamus/devops-team
*.toml @price-stradamus/core-team

# Security-sensitive files require security review
**/auth*.py @price-stradamus/security-team
**/security*.py @price-stradamus/security-team
.env.example @price-stradamus/security-team
```

### Code Ownership Guidelines

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class CodeOwnership:
    """Code ownership rules and guidelines."""

    # Ownership matrix
    OWNERSHIP_MATRIX = {
        "data/": {
            "primary": "data-team",
            "secondary": "ml-team",
            "review_required": 1,
            "expertise": ["PostgreSQL", "asyncpg", "data pipelines"],
        },
        "models/": {
            "primary": "ml-team",
            "secondary": "quant-team",
            "review_required": 2,
            "expertise": ["PyTorch", "Darts", "time series"],
        },
        "evaluation/": {
            "primary": "ml-team",
            "secondary": "quant-team",
            "review_required": 2,
            "expertise": ["backtesting", "metrics", "statistics"],
        },
        "config/": {
            "primary": "core-team",
            "secondary": "devops-team",
            "review_required": 1,
            "expertise": ["pydantic", "configuration management"],
        },
    }

    @classmethod
    def get_owners(cls, file_path: str) -> dict:
        """Get owners for a file path.

        Args:
            file_path: Relative file path

        Returns:
            Ownership information
        """
        path = Path(file_path)
        for pattern, info in cls.OWNERSHIP_MATRIX.items():
            if pattern.rstrip("/") in str(path):
                return info
        return {
            "primary": "core-team",
            "secondary": None,
            "review_required": 1,
            "expertise": [],
        }

    @classmethod
    def format_review_requirements(cls, file_path: str) -> str:
        """Format review requirements for PR description.

        Args:
            file_path: Changed file path

        Returns:
            Formatted review requirements
        """
        owners = cls.get_owners(file_path)
        return (
            f"File: {file_path}\n"
            f"Primary Owner: @price-stradamus/{owners['primary']}\n"
            f"Reviews Required: {owners['review_required']}\n"
            f"Expertise Needed: {', '.join(owners['expertise'])}"
        )
```

---

## Breaking Change Policy [PHASE 3-4]

### Semantic Versioning

Follow **Semantic Versioning 2.0.0** (SemVer):
- **MAJOR**: Breaking changes (X.0.0)
- **MINOR**: New features, backward compatible (0.X.0)
- **PATCH**: Bug fixes, backward compatible (0.0.X)

### Breaking Change Detection

```python
from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class ChangeType(str, Enum):
    """Type of API change."""

    BREAKING = "breaking"
    DEPRECATION = "deprecation"
    ADDITION = "addition"
    COMPATIBLE = "compatible"


@dataclass
class APISignature:
    """Captured API signature for comparison."""

    name: str
    parameters: list[tuple[str, Any, Any]]  # (name, annotation, default)
    return_annotation: Any
    docstring: str | None


@dataclass
class APIChange:
    """Detected API change."""

    change_type: ChangeType
    location: str
    description: str
    migration_guide: str | None = None


class BreakingChangeDetector:
    """Detect breaking changes between API versions."""

    BREAKING_CHANGES = {
        "removed_function": "Function '{name}' was removed",
        "removed_class": "Class '{name}' was removed",
        "removed_parameter": "Parameter '{param}' removed from '{name}'",
        "required_parameter_added": "Required parameter '{param}' added to '{name}'",
        "return_type_changed": "Return type of '{name}' changed from {old} to {new}",
        "parameter_type_changed": "Type of '{param}' in '{name}' changed from {old} to {new}",
    }

    def compare_signatures(
        self,
        old_sig: APISignature,
        new_sig: APISignature,
    ) -> list[APIChange]:
        """Compare API signatures for breaking changes.

        Args:
            old_sig: Previous API signature
            new_sig: New API signature

        Returns:
            List of detected changes
        """
        changes = []

        old_params = {p[0]: p for p in old_sig.parameters}
        new_params = {p[0]: p for p in new_sig.parameters}

        # Check for removed parameters
        for name in old_params:
            if name not in new_params:
                changes.append(APIChange(
                    change_type=ChangeType.BREAKING,
                    location=old_sig.name,
                    description=self.BREAKING_CHANGES["removed_parameter"].format(
                        param=name, name=old_sig.name
                    ),
                    migration_guide=f"Remove usage of parameter '{name}'",
                ))

        # Check for new required parameters
        for name, (_, annotation, default) in new_params.items():
            if name not in old_params and default is inspect.Parameter.empty:
                changes.append(APIChange(
                    change_type=ChangeType.BREAKING,
                    location=new_sig.name,
                    description=self.BREAKING_CHANGES["required_parameter_added"].format(
                        param=name, name=new_sig.name
                    ),
                    migration_guide=f"Add required parameter '{name}' to all calls",
                ))

        # Check return type changes
        if old_sig.return_annotation != new_sig.return_annotation:
            changes.append(APIChange(
                change_type=ChangeType.BREAKING,
                location=new_sig.name,
                description=self.BREAKING_CHANGES["return_type_changed"].format(
                    name=new_sig.name,
                    old=old_sig.return_annotation,
                    new=new_sig.return_annotation,
                ),
                migration_guide="Update code to handle new return type",
            ))

        return changes

    def generate_changelog_entry(self, changes: list[APIChange]) -> str:
        """Generate changelog entry for breaking changes.

        Args:
            changes: List of API changes

        Returns:
            Formatted changelog entry
        """
        breaking = [c for c in changes if c.change_type == ChangeType.BREAKING]
        if not breaking:
            return ""

        lines = ["## ⚠️ Breaking Changes\n"]
        for change in breaking:
            lines.append(f"### {change.location}")
            lines.append(f"- {change.description}")
            if change.migration_guide:
                lines.append(f"- **Migration**: {change.migration_guide}")
            lines.append("")

        return "\n".join(lines)
```

### Breaking Change Checklist

Before making breaking changes:

1. **Document the change** in CHANGELOG.md
2. **Provide migration guide** with code examples
3. **Deprecate first** (one minor version minimum)
4. **Update all documentation**
5. **Notify users** via release notes
6. **Provide automated migration** tools when possible

---

## Deprecation Policy [PHASE 3-4]

### Deprecation Decorator

```python
from __future__ import annotations

import functools
import warnings
from datetime import date
from typing import Callable, TypeVar

from loguru import logger

F = TypeVar("F", bound=Callable)


def deprecated(
    reason: str,
    removal_version: str,
    replacement: str | None = None,
    removal_date: date | None = None,
) -> Callable[[F], F]:
    """Mark a function as deprecated.

    Args:
        reason: Why the function is deprecated
        removal_version: Version when it will be removed
        replacement: Suggested replacement function
        removal_date: Planned removal date

    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            message = (
                f"{func.__name__} is deprecated: {reason}. "
                f"Will be removed in version {removal_version}."
            )
            if replacement:
                message += f" Use {replacement} instead."
            if removal_date:
                message += f" Planned removal: {removal_date.isoformat()}."

            warnings.warn(message, DeprecationWarning, stacklevel=2)
            logger.warning(
                "Deprecated function called",
                function=func.__name__,
                replacement=replacement,
                removal_version=removal_version,
            )
            return func(*args, **kwargs)

        # Add deprecation info to docstring
        deprecation_note = f"""
        .. deprecated:: {removal_version}
           {reason}
           {"Use :func:`" + replacement + "` instead." if replacement else ""}
        """
        wrapper.__doc__ = (wrapper.__doc__ or "") + deprecation_note

        return wrapper  # type: ignore

    return decorator


class DeprecatedClass:
    """Mixin for deprecated classes."""

    _deprecation_warning_shown = False

    def __init_subclass__(
        cls,
        deprecation_reason: str = "",
        removal_version: str = "",
        replacement: str | None = None,
        **kwargs,
    ):
        super().__init_subclass__(**kwargs)
        cls._deprecation_reason = deprecation_reason
        cls._removal_version = removal_version
        cls._replacement = replacement

    def __new__(cls, *args, **kwargs):
        if not cls._deprecation_warning_shown and hasattr(cls, "_deprecation_reason"):
            message = (
                f"{cls.__name__} is deprecated: {cls._deprecation_reason}. "
                f"Will be removed in version {cls._removal_version}."
            )
            if cls._replacement:
                message += f" Use {cls._replacement} instead."
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            cls._deprecation_warning_shown = True
        return super().__new__(cls)


# Usage examples
@deprecated(
    reason="This function uses inefficient algorithm",
    removal_version="2.0.0",
    replacement="calculate_rsi_vectorized",
    removal_date=date(2025, 6, 1),
)
def calculate_rsi_legacy(prices: list[float], period: int = 14) -> list[float]:
    """Calculate RSI using legacy method."""
    # Old implementation
    return []


def calculate_rsi_vectorized(prices: list[float], period: int = 14) -> list[float]:
    """Calculate RSI using vectorized operations (recommended)."""
    # New, faster implementation
    return []


# Deprecated class example
class OldPreprocessor(
    DeprecatedClass,
    deprecation_reason="Uses outdated normalization",
    removal_version="2.0.0",
    replacement="ModernPreprocessor",
):
    """Old preprocessing class."""
    pass
```

### Deprecation Timeline

```
┌────────────────────────────────────────────────────────────────┐
│                    Deprecation Timeline                         │
├────────────────────────────────────────────────────────────────┤
│  v1.5.0          v1.6.0          v1.7.0          v2.0.0       │
│     │               │               │               │          │
│     │  Deprecation  │  Warning in   │  Warning in   │  Removal │
│     │  announced    │  logs only    │  all calls    │          │
│     │               │               │               │          │
│     ├───────────────┼───────────────┼───────────────┤          │
│     │    30 days    │    30 days    │    30 days    │          │
│     │               │               │               │          │
│     │  Add @deprecated decorator                    │          │
│     │  Update docs with deprecation notice          │          │
│     │  Add migration guide                          │          │
└────────────────────────────────────────────────────────────────┘
```

---

## Performance Testing Guidelines [PHASE 3]

### Benchmark Framework

```python
from __future__ import annotations

import gc
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable

import psutil
from loguru import logger


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    name: str
    iterations: int
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    memory_peak_mb: float
    throughput: float | None = None

    def __str__(self) -> str:
        return (
            f"{self.name}:\n"
            f"  Mean: {self.mean_time_ms:.2f}ms ± {self.std_time_ms:.2f}ms\n"
            f"  Min: {self.min_time_ms:.2f}ms, Max: {self.max_time_ms:.2f}ms\n"
            f"  Memory Peak: {self.memory_peak_mb:.1f}MB\n"
            f"  Throughput: {self.throughput:.1f}/s" if self.throughput else ""
        )


@dataclass
class PerformanceBudget:
    """Performance budget thresholds."""

    max_latency_ms: float
    max_memory_mb: float
    min_throughput: float | None = None


class PerformanceBenchmark:
    """Performance benchmarking framework."""

    def __init__(
        self,
        warmup_iterations: int = 3,
        benchmark_iterations: int = 10,
    ) -> None:
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations

    def benchmark(
        self,
        name: str,
        func: Callable[[], None],
        setup: Callable[[], None] | None = None,
        teardown: Callable[[], None] | None = None,
    ) -> BenchmarkResult:
        """Run benchmark on a function.

        Args:
            name: Benchmark name
            func: Function to benchmark
            setup: Optional setup function
            teardown: Optional teardown function

        Returns:
            Benchmark results
        """
        times_ms: list[float] = []
        memory_peak = 0.0
        process = psutil.Process()

        # Warmup
        for _ in range(self.warmup_iterations):
            if setup:
                setup()
            func()
            if teardown:
                teardown()

        # Force garbage collection before benchmark
        gc.collect()

        # Benchmark runs
        for _ in range(self.benchmark_iterations):
            if setup:
                setup()

            gc.disable()
            memory_before = process.memory_info().rss

            start = time.perf_counter()
            func()
            end = time.perf_counter()

            memory_after = process.memory_info().rss
            gc.enable()

            times_ms.append((end - start) * 1000)
            memory_peak = max(memory_peak, (memory_after - memory_before) / 1024 / 1024)

            if teardown:
                teardown()

        return BenchmarkResult(
            name=name,
            iterations=self.benchmark_iterations,
            mean_time_ms=statistics.mean(times_ms),
            std_time_ms=statistics.stdev(times_ms) if len(times_ms) > 1 else 0,
            min_time_ms=min(times_ms),
            max_time_ms=max(times_ms),
            memory_peak_mb=memory_peak,
            throughput=1000 / statistics.mean(times_ms),  # ops/sec
        )

    def check_budget(
        self,
        result: BenchmarkResult,
        budget: PerformanceBudget,
    ) -> tuple[bool, list[str]]:
        """Check if benchmark result meets performance budget.

        Args:
            result: Benchmark result
            budget: Performance budget

        Returns:
            Tuple of (passed, list of violations)
        """
        violations = []

        if result.mean_time_ms > budget.max_latency_ms:
            violations.append(
                f"Latency {result.mean_time_ms:.2f}ms exceeds budget "
                f"{budget.max_latency_ms:.2f}ms"
            )

        if result.memory_peak_mb > budget.max_memory_mb:
            violations.append(
                f"Memory {result.memory_peak_mb:.1f}MB exceeds budget "
                f"{budget.max_memory_mb:.1f}MB"
            )

        if budget.min_throughput and result.throughput:
            if result.throughput < budget.min_throughput:
                violations.append(
                    f"Throughput {result.throughput:.1f}/s below minimum "
                    f"{budget.min_throughput:.1f}/s"
                )

        return len(violations) == 0, violations


# Performance budgets for Price Stradamus
PERFORMANCE_BUDGETS = {
    "prediction_inference": PerformanceBudget(
        max_latency_ms=100,
        max_memory_mb=500,
        min_throughput=10,
    ),
    "feature_generation": PerformanceBudget(
        max_latency_ms=500,
        max_memory_mb=1000,
        min_throughput=5,
    ),
    "data_fetch": PerformanceBudget(
        max_latency_ms=2000,
        max_memory_mb=200,
    ),
    "model_training": PerformanceBudget(
        max_latency_ms=300000,  # 5 minutes
        max_memory_mb=8000,
    ),
}


# Usage in tests
def test_prediction_performance():
    """Test prediction meets performance budget."""
    benchmark = PerformanceBenchmark()

    def predict():
        # Your prediction code here
        pass

    result = benchmark.benchmark("prediction_inference", predict)
    passed, violations = benchmark.check_budget(
        result,
        PERFORMANCE_BUDGETS["prediction_inference"],
    )

    assert passed, f"Performance budget violated: {violations}"
```

### Load Testing

```python
from __future__ import annotations

import asyncio
import statistics
import time
from dataclasses import dataclass


@dataclass
class LoadTestResult:
    """Results from load test."""

    total_requests: int
    successful_requests: int
    failed_requests: int
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    requests_per_second: float
    duration_seconds: float


class LoadTester:
    """Simple async load tester."""

    def __init__(
        self,
        concurrent_users: int = 10,
        duration_seconds: int = 30,
    ) -> None:
        self.concurrent_users = concurrent_users
        self.duration_seconds = duration_seconds

    async def run(
        self,
        request_func: Callable[[], Awaitable[bool]],
    ) -> LoadTestResult:
        """Run load test.

        Args:
            request_func: Async function returning True on success

        Returns:
            Load test results
        """
        latencies: list[float] = []
        successes = 0
        failures = 0
        start_time = time.monotonic()

        async def worker():
            nonlocal successes, failures
            while time.monotonic() - start_time < self.duration_seconds:
                req_start = time.perf_counter()
                try:
                    success = await request_func()
                    if success:
                        successes += 1
                    else:
                        failures += 1
                except Exception:
                    failures += 1
                req_end = time.perf_counter()
                latencies.append((req_end - req_start) * 1000)

        # Run concurrent workers
        await asyncio.gather(*[
            worker() for _ in range(self.concurrent_users)
        ])

        end_time = time.monotonic()
        duration = end_time - start_time

        sorted_latencies = sorted(latencies)
        total = len(latencies)

        return LoadTestResult(
            total_requests=total,
            successful_requests=successes,
            failed_requests=failures,
            mean_latency_ms=statistics.mean(latencies),
            p50_latency_ms=sorted_latencies[int(total * 0.50)],
            p95_latency_ms=sorted_latencies[int(total * 0.95)],
            p99_latency_ms=sorted_latencies[int(total * 0.99)],
            requests_per_second=total / duration,
            duration_seconds=duration,
        )
```

---

## Structured Concurrency (TaskGroup) [PHASE 2-3]

### Python 3.11+ TaskGroup Patterns

Use `asyncio.TaskGroup` for structured concurrency instead of bare `asyncio.gather()`.

```python
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from loguru import logger


@dataclass
class FetchResult:
    """Result from data fetch operation."""

    symbol: str
    data: Any
    success: bool
    error: str | None = None


class DataFetcherWithTaskGroup:
    """Data fetcher using structured concurrency."""

    def __init__(self, max_concurrent: int = 5) -> None:
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def fetch_symbol(self, symbol: str) -> FetchResult:
        """Fetch data for a single symbol with rate limiting.

        Args:
            symbol: Trading symbol

        Returns:
            Fetch result
        """
        async with self.semaphore:
            try:
                # Simulated fetch
                await asyncio.sleep(0.1)
                data = {"symbol": symbol, "price": 50000}
                return FetchResult(symbol=symbol, data=data, success=True)
            except Exception as e:
                logger.error("Fetch failed for {symbol}: {error}", symbol=symbol, error=e)
                return FetchResult(symbol=symbol, data=None, success=False, error=str(e))

    async def fetch_multiple_symbols(
        self,
        symbols: list[str],
    ) -> list[FetchResult]:
        """Fetch data for multiple symbols using TaskGroup.

        TaskGroup advantages over gather():
        1. Automatic cancellation of remaining tasks on first exception
        2. Proper cleanup of resources
        3. All exceptions are collected and raised together
        4. More explicit error handling

        Args:
            symbols: List of trading symbols

        Returns:
            List of fetch results
        """
        results: list[FetchResult] = []

        async with asyncio.TaskGroup() as tg:
            tasks = [
                tg.create_task(self.fetch_symbol(symbol))
                for symbol in symbols
            ]

        # All tasks completed successfully if we reach here
        results = [task.result() for task in tasks]
        return results

    async def fetch_with_timeout(
        self,
        symbols: list[str],
        timeout_seconds: float = 30.0,
    ) -> list[FetchResult]:
        """Fetch with overall timeout using TaskGroup.

        Args:
            symbols: Symbols to fetch
            timeout_seconds: Maximum time for all fetches

        Returns:
            Fetch results (may be partial on timeout)

        Raises:
            TimeoutError: If timeout exceeded
        """
        try:
            async with asyncio.timeout(timeout_seconds):
                return await self.fetch_multiple_symbols(symbols)
        except TimeoutError:
            logger.warning(
                "Fetch timed out after {timeout}s",
                timeout=timeout_seconds,
            )
            raise


class PipelineWithTaskGroup:
    """ML pipeline using structured concurrency."""

    async def run_parallel_preprocessing(
        self,
        datasets: list[str],
    ) -> dict[str, Any]:
        """Run preprocessing on multiple datasets in parallel.

        Args:
            datasets: Dataset identifiers

        Returns:
            Preprocessed data by dataset
        """
        results: dict[str, Any] = {}

        async def preprocess_dataset(name: str) -> tuple[str, Any]:
            """Preprocess single dataset."""
            await asyncio.sleep(0.5)  # Simulated work
            return name, {"preprocessed": True}

        async with asyncio.TaskGroup() as tg:
            tasks = [
                tg.create_task(preprocess_dataset(name))
                for name in datasets
            ]

        for task in tasks:
            name, data = task.result()
            results[name] = data

        return results

    async def run_pipeline_stages(self) -> dict[str, Any]:
        """Run pipeline with dependent stages.

        Stage 1: Fetch data (parallel)
        Stage 2: Preprocess (parallel)
        Stage 3: Generate features (parallel)
        Stage 4: Train model (sequential)
        """
        # Stage 1: Parallel data fetch
        async with asyncio.TaskGroup() as tg:
            fetch_task = tg.create_task(self._fetch_data())
            validate_task = tg.create_task(self._validate_config())

        data = fetch_task.result()
        config = validate_task.result()

        # Stage 2: Parallel preprocessing
        async with asyncio.TaskGroup() as tg:
            clean_task = tg.create_task(self._clean_data(data))
            normalize_task = tg.create_task(self._prepare_normalization(data))

        clean_data = clean_task.result()
        normalizer = normalize_task.result()

        # Stage 3: Feature generation (parallel)
        async with asyncio.TaskGroup() as tg:
            features_tasks = [
                tg.create_task(self._generate_feature_group(clean_data, group))
                for group in ["price", "volume", "momentum"]
            ]

        features = {}
        for task in features_tasks:
            features.update(task.result())

        # Stage 4: Training (sequential, GPU-bound)
        model = await self._train_model(features, config)

        return {"model": model, "features": features}

    async def _fetch_data(self) -> dict:
        await asyncio.sleep(0.1)
        return {"raw": "data"}

    async def _validate_config(self) -> dict:
        await asyncio.sleep(0.05)
        return {"validated": True}

    async def _clean_data(self, data: dict) -> dict:
        await asyncio.sleep(0.1)
        return {"cleaned": data}

    async def _prepare_normalization(self, data: dict) -> dict:
        await asyncio.sleep(0.1)
        return {"normalizer": "ready"}

    async def _generate_feature_group(self, data: dict, group: str) -> dict:
        await asyncio.sleep(0.1)
        return {group: f"{group}_features"}

    async def _train_model(self, features: dict, config: dict) -> str:
        await asyncio.sleep(0.2)
        return "trained_model"


class GracefulShutdownHandler:
    """Handle graceful shutdown with TaskGroup cleanup."""

    def __init__(self) -> None:
        self._shutdown_event = asyncio.Event()
        self._running_tasks: set[asyncio.Task] = set()

    async def run_with_shutdown(
        self,
        main_coro: Coroutine,
    ) -> Any:
        """Run coroutine with graceful shutdown handling.

        Args:
            main_coro: Main coroutine to run

        Returns:
            Result from main coroutine
        """
        main_task = asyncio.create_task(main_coro)
        shutdown_task = asyncio.create_task(self._shutdown_event.wait())

        done, pending = await asyncio.wait(
            {main_task, shutdown_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Cancel pending tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        if main_task in done:
            return main_task.result()

        raise asyncio.CancelledError("Shutdown requested")

    def request_shutdown(self) -> None:
        """Signal shutdown."""
        self._shutdown_event.set()


# Example usage with exception handling
async def robust_parallel_operation():
    """Demonstrate robust parallel operations with TaskGroup."""
    fetcher = DataFetcherWithTaskGroup()

    try:
        # TaskGroup will cancel all tasks if any raises
        results = await fetcher.fetch_multiple_symbols([
            "BTCUSDT", "ETHUSDT", "SOLUSDT"
        ])
        logger.info("All fetches completed", count=len(results))
        return results

    except* ConnectionError as eg:
        # Python 3.11+ except* syntax for ExceptionGroups
        logger.error("Connection errors occurred", errors=eg.exceptions)
        raise

    except* asyncio.TimeoutError as eg:
        logger.error("Timeout errors occurred", errors=eg.exceptions)
        raise
```

### Migration from gather() to TaskGroup

```python
# BEFORE: Using asyncio.gather()
async def old_way():
    tasks = [fetch(s) for s in symbols]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    # Must manually check for exceptions in results
    for r in results:
        if isinstance(r, Exception):
            handle_error(r)

# AFTER: Using asyncio.TaskGroup (Python 3.11+)
async def new_way():
    try:
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(fetch(s)) for s in symbols]
        # All tasks succeeded
        results = [t.result() for t in tasks]
    except* Exception as eg:
        # Handle exception group
        for exc in eg.exceptions:
            handle_error(exc)
```

---

## CLI Accessibility Guidelines [PHASE 4]

### Accessible Output Formatting

```python
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from enum import Enum
from typing import TextIO

from rich.console import Console
from rich.table import Table
from rich.theme import Theme


class OutputFormat(str, Enum):
    """Supported output formats."""

    RICH = "rich"  # Rich formatted (default)
    PLAIN = "plain"  # Plain text (accessibility)
    JSON = "json"  # Machine readable
    CSV = "csv"  # Spreadsheet compatible


@dataclass
class AccessibilitySettings:
    """Accessibility configuration."""

    # Detect from environment
    no_color: bool = bool(os.environ.get("NO_COLOR"))
    force_color: bool = bool(os.environ.get("FORCE_COLOR"))
    term_program: str = os.environ.get("TERM_PROGRAM", "")

    # Screen reader mode
    screen_reader: bool = bool(os.environ.get("SCREEN_READER"))

    # Reduced motion
    reduce_motion: bool = bool(os.environ.get("REDUCE_MOTION"))

    @property
    def use_color(self) -> bool:
        """Determine if color should be used."""
        if self.no_color:
            return False
        if self.force_color:
            return True
        return sys.stdout.isatty()

    @property
    def use_animation(self) -> bool:
        """Determine if animations should be used."""
        return not self.reduce_motion and sys.stdout.isatty()


class AccessibleCLI:
    """CLI with accessibility features."""

    # High contrast theme for visibility
    HIGH_CONTRAST_THEME = Theme({
        "info": "bright_white",
        "warning": "bright_yellow bold",
        "error": "bright_red bold",
        "success": "bright_green bold",
        "highlight": "bright_cyan bold",
        "muted": "white",
    })

    def __init__(
        self,
        output_format: OutputFormat = OutputFormat.RICH,
        stream: TextIO = sys.stdout,
    ) -> None:
        self.settings = AccessibilitySettings()
        self.output_format = output_format
        self.stream = stream

        # Configure Rich console
        self.console = Console(
            file=stream,
            force_terminal=self.settings.use_color,
            no_color=not self.settings.use_color,
            theme=self.HIGH_CONTRAST_THEME,
        )

    def print_table(
        self,
        title: str,
        headers: list[str],
        rows: list[list[str]],
    ) -> None:
        """Print table in accessible format.

        Args:
            title: Table title
            headers: Column headers
            rows: Table data
        """
        match self.output_format:
            case OutputFormat.RICH:
                self._print_rich_table(title, headers, rows)
            case OutputFormat.PLAIN:
                self._print_plain_table(title, headers, rows)
            case OutputFormat.JSON:
                self._print_json_table(title, headers, rows)
            case OutputFormat.CSV:
                self._print_csv_table(headers, rows)

    def _print_rich_table(
        self,
        title: str,
        headers: list[str],
        rows: list[list[str]],
    ) -> None:
        """Print Rich formatted table."""
        table = Table(title=title, show_header=True, header_style="bold")

        for header in headers:
            table.add_column(header)

        for row in rows:
            table.add_row(*row)

        self.console.print(table)

    def _print_plain_table(
        self,
        title: str,
        headers: list[str],
        rows: list[list[str]],
    ) -> None:
        """Print plain text table (screen reader friendly)."""
        # Calculate column widths
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))

        # Print title
        print(f"\n{title}", file=self.stream)
        print("=" * len(title), file=self.stream)

        # Print header
        header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
        print(header_line, file=self.stream)
        print("-" * len(header_line), file=self.stream)

        # Print rows
        for row in rows:
            row_line = " | ".join(
                cell.ljust(widths[i]) for i, cell in enumerate(row)
            )
            print(row_line, file=self.stream)

        # Summary for screen readers
        print(f"\n{len(rows)} rows displayed.", file=self.stream)

    def _print_json_table(
        self,
        title: str,
        headers: list[str],
        rows: list[list[str]],
    ) -> None:
        """Print JSON formatted output."""
        import json

        data = {
            "title": title,
            "headers": headers,
            "rows": [dict(zip(headers, row)) for row in rows],
            "count": len(rows),
        }
        print(json.dumps(data, indent=2), file=self.stream)

    def _print_csv_table(
        self,
        headers: list[str],
        rows: list[list[str]],
    ) -> None:
        """Print CSV formatted output."""
        import csv
        import io

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(headers)
        writer.writerows(rows)
        print(output.getvalue(), file=self.stream)

    def print_progress(
        self,
        current: int,
        total: int,
        description: str,
    ) -> None:
        """Print progress in accessible format.

        Args:
            current: Current progress
            total: Total items
            description: Progress description
        """
        percentage = (current / total) * 100 if total > 0 else 0

        if self.settings.screen_reader or self.output_format == OutputFormat.PLAIN:
            # Screen reader friendly: announce milestones
            if current == total:
                print(f"{description}: Complete ({total} items)", file=self.stream)
            elif percentage % 25 == 0:  # Announce at 25%, 50%, 75%
                print(f"{description}: {percentage:.0f}% ({current}/{total})", file=self.stream)
        elif self.settings.use_animation:
            # Rich progress bar
            from rich.progress import Progress
            # Use Rich progress
            pass
        else:
            # Static progress
            bar_width = 30
            filled = int(bar_width * current / total)
            bar = "█" * filled + "░" * (bar_width - filled)
            print(f"\r{description}: [{bar}] {percentage:.0f}%", end="", file=self.stream)
            if current == total:
                print(file=self.stream)

    def print_status(
        self,
        message: str,
        status: str = "info",
    ) -> None:
        """Print status message with appropriate styling.

        Args:
            message: Status message
            status: Status type (info, warning, error, success)
        """
        # Status symbols (text alternatives for screen readers)
        symbols = {
            "info": ("ℹ", "INFO"),
            "warning": ("⚠", "WARNING"),
            "error": ("✗", "ERROR"),
            "success": ("✓", "SUCCESS"),
        }

        symbol, text_alt = symbols.get(status, ("•", status.upper()))

        if self.settings.screen_reader or self.output_format == OutputFormat.PLAIN:
            print(f"[{text_alt}] {message}", file=self.stream)
        else:
            self.console.print(f"[{status}]{symbol} {message}[/{status}]")


# Usage with Typer
import typer

app = typer.Typer()

@app.command()
def predictions(
    format: OutputFormat = typer.Option(
        OutputFormat.RICH,
        "--format", "-f",
        help="Output format (rich, plain, json, csv)",
    ),
):
    """Show recent predictions."""
    cli = AccessibleCLI(output_format=format)

    cli.print_table(
        title="Recent Predictions",
        headers=["Timestamp", "Model", "Prediction", "Actual", "Error"],
        rows=[
            ["2025-01-01 12:00", "nbeats", "50100", "50050", "-0.1%"],
            ["2025-01-01 12:01", "nbeats", "50120", "50080", "-0.08%"],
        ],
    )
```

---

## Resources [ALL PHASES]

### Documentation
- [Darts Documentation](https://unit8co.github.io/darts/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [pandas-ta Documentation](https://github.com/twopirllc/pandas-ta)
- [Binance API Documentation](https://binance-docs.github.io/apidocs/)

### Papers
