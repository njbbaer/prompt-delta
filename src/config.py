from dataclasses import dataclass, field
from typing import Dict, List
import jinja2

from .yaml_config import yaml


@dataclass
class Config:
    comparison_prompt: str
    content_variations: Dict[str, str]
    content_prompts: Dict[str, str] = field(default_factory=dict)
    tags: List[str] = None
    warm_cache: bool = False
    model: str = "anthropic/claude-3.5-haiku:beta"
    batch_size: int = 1
    max_retries: int = 3

    @classmethod
    async def load(cls, filepath) -> "Config":
        with open(filepath, "r") as f:
            data = yaml.load(f)
        resolved_data = cls._resolve_vars(data)

        return cls(**resolved_data)

    @property
    def total_calls(self) -> int:
        return len(self.content_prompts) * self.batch_size

    @staticmethod
    def _resolve_vars(obj):
        env = jinja2.Environment(loader=jinja2.FileSystemLoader("."), autoescape=False)

        def load_file(filename):
            template = env.get_template(filename)
            return template.render()

        env.globals["load"] = load_file

        for _ in range(10):
            resolved = Config._resolve_vars_recursive(obj, env)
            if resolved == obj:
                return resolved
            obj = resolved

        raise RuntimeError("Too many iterations resolving vars. Circular reference?")

    @staticmethod
    def _resolve_vars_recursive(obj, env):
        if isinstance(obj, dict):
            return {k: Config._resolve_vars_recursive(v, env) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [Config._resolve_vars_recursive(i, env) for i in obj]
        elif isinstance(obj, str):
            template = env.from_string(str(obj))
            return template.render()
        return obj
