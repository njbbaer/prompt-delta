import io
import os
from datetime import datetime

from ruamel.yaml.scalarstring import LiteralScalarString

from .yaml_config import yaml


class Logger:
    def __init__(self, filepath):
        self.filepath = filepath

    def log(self, id, cost, cache_discount, params, response):
        buffer = io.StringIO()
        params["messages"] = self._format_text(params["messages"])
        yaml.dump(
            [
                {
                    "timestamp": self._current_timestamp(),
                    "id": id,
                    "cost": cost,
                    "cache_discount": cache_discount,
                    "parameters": params,
                    "response": LiteralScalarString(response),
                }
            ],
            buffer,
        )

        self._ensure_directory_exists()

        with open(self.filepath, "a") as file:
            file.write(buffer.getvalue())

    def _ensure_directory_exists(self):
        directory = os.path.dirname(self.filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

    @staticmethod
    def _current_timestamp():
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _format_text(data):
        if isinstance(data, dict):
            return {
                k: (
                    LiteralScalarString(v)
                    if k == "text" or (k == "content" and isinstance(v, str))
                    else Logger._format_text(v)
                )
                for k, v in data.items()
            }
        elif isinstance(data, list):
            return [Logger._format_text(item) for item in data]
        else:
            return data
