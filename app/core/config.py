"""Core Configuration
Re-exports settings from app.config.settings for backward compatibility.
"""

from app.config.settings import settings, Settings

__all__ = ["settings", "Settings"]

