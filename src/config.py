from dataclasses import dataclass, field
from typing import Dict, List, Any
import jinja2
import os

from .yaml_config import yaml


@dataclass(kw_only=True)
class Config:
    config_key: str = ""
    timeout: int = 60
    max_retries: int = 3
    iterations: int = 1
    model: str
    log_file: str
    output_file: str

    @classmethod
    def load(cls, filepath: str) -> "Config":
        with open(filepath, "r") as f:
            full_data = yaml.load(f)

        # Set working directory to the config file directory
        os.chdir(os.path.dirname(filepath))

        subtype_data = full_data.get("shared", {}) | full_data[cls.config_key]
        resolved_data = cls._resolve_vars(subtype_data)

        return cls(**resolved_data)

    @staticmethod
    def _resolve_vars(obj: Any) -> Any:
        env = jinja2.Environment(loader=jinja2.FileSystemLoader("."), autoescape=False)

        def load_file(filename: str) -> str:
            template = env.get_template(filename)
            return template.render()

        env.globals["load"] = load_file

        for _ in range(10):
            resolved = Config._resolve_vars_recursive(obj, env)
            if resolved == obj:
                return resolved
            obj = resolved

        raise RuntimeError("Unable to resolve config data")

    @staticmethod
    def _resolve_vars_recursive(obj: Any, env: jinja2.Environment) -> Any:
        if isinstance(obj, dict):
            return {k: Config._resolve_vars_recursive(v, env) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [Config._resolve_vars_recursive(i, env) for i in obj]
        elif isinstance(obj, str):
            template = env.from_string(str(obj))
            return template.render()
        return obj

    @property
    def api_key(self):
        return os.getenv("OPENROUTER_API_KEY")


@dataclass(kw_only=True)
class GenerationConfig(Config):
    config_key: str = "generation"
    content_prompts: Dict[str, str] = field(default_factory=dict)
    response_tags: List[str] = []
    warm_cache: bool = False
    content_variations: Dict[str, str]
