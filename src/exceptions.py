"""Application-level exceptions shown as clean CLI errors."""


class ResearchAssistantError(RuntimeError):
    """Base class for expected application failures."""


class InvalidQuestionError(ResearchAssistantError):
    """Raised when a research question fails validation."""


class ExternalServiceError(ResearchAssistantError):
    """Raised after an upstream call exhausts its retry budget."""


class InvalidAIResponseError(ResearchAssistantError):
    """Raised when an upstream response is structurally or semantically invalid."""


class NoSourcesError(ResearchAssistantError):
    """Raised when every selected source is unavailable or empty."""


class CacheError(ResearchAssistantError):
    """Raised when the persistent source cache cannot be read or written."""


class StorageError(ResearchAssistantError):
    """Raised when research history cannot be accessed."""

