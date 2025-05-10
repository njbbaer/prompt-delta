import jinja2
import os
from typing import Any

from .yaml_config import yaml


class Config:
    def __init__(self, filepath: str, subtype: str) -> None:
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.timeout = 60
        self.max_retries = 3

        self._load(filepath, subtype)

    def _load(self, filepath: str, subtype: str) -> None:
        with open(filepath, "r") as f:
            full_data = yaml.load(f)

        # Set working directory to the config file directory
        os.chdir(os.path.dirname(filepath))

        subtype_data = full_data.get("shared", {}) | full_data[subtype]
        resolved_data = self._resolve_vars(subtype_data)

        for key, value in resolved_data.items():
            setattr(self, key, value)

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
