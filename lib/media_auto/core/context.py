from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass(frozen=True)
class GenerationContext:
    """
    Immutable context object holding all configuration for a generation task.
    Replaces the mutable GenerationConfig.
    """
    character: str
    prompt: str
    output_dir: str
    # Additional dynamic attributes can be stored in a dictionary if needed,
    # but explicit fields are preferred for clarity.
    extra_config: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Safe accessor for extra config."""
        return self.extra_config.get(key, default)
