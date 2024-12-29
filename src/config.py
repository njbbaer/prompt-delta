from dataclasses import dataclass
from typing import Dict

from .yaml_config import yaml


@dataclass
class Config:
    content_prompts: Dict[str, str]
    comparison_prompt: str
    model: str = "anthropic/claude-3.5-haiku:beta"
    batch_size: int = 1
    max_retries: int = 3

    @classmethod
    async def load(cls, filepath) -> "Config":
        with open(filepath, "r") as f:
            data = yaml.load(f)
        return cls(**data)

    @property
    def total_calls(self) -> int:
        return len(self.content_prompts) * self.batch_size
