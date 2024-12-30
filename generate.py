import asyncio
import re
from typing import Dict, List, Tuple, AsyncIterator
from pathlib import Path
from tqdm import tqdm
from ruamel.yaml.scalarstring import LiteralScalarString

from src.api_client import OpenRouterClient
from src.config import Config
from src.yaml_config import yaml


async def make_request(client: OpenRouterClient, prompt: str, config: Config) -> str:
    response = await client.request_chat_completion(
        {
            "model": config.model,
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                },
                {"role": "user", "content": config.content_message},
            ],
        }
    )
    return strip_tags(response, config.strip_tags)


async def execute_batch(
    tasks: List[Tuple[str, AsyncIterator]], pbar: tqdm
) -> Dict[str, List[str]]:
    results = {}
    completed = await asyncio.gather(*[task for _, task in tasks])
    pbar.update(len(completed))

    for (name, _), result in zip(tasks, completed):
        results.setdefault(name, []).append(LiteralScalarString(result))

    return results


async def process_prompts(client: OpenRouterClient, config: Config) -> Dict:
    with tqdm(total=config.total_calls, desc="Processing") as pbar:
        results = {}

        if config.warm_cache:
            warm_tasks = [
                (name, make_request(client, prompt, config))
                for name, prompt in config.content_prompts.items()
            ]
            results = await execute_batch(warm_tasks, pbar)

        remaining_size = config.batch_size - (1 if config.warm_cache else 0)
        remaining_tasks = [
            (name, make_request(client, prompt, config))
            for name, prompt in config.content_prompts.items()
            for _ in range(remaining_size)
        ]

        remaining_results = await execute_batch(remaining_tasks, pbar)

        for name, responses in remaining_results.items():
            results.setdefault(name, []).extend(responses)

    return results


def strip_tags(content: str, tags: List[str]) -> str:
    for tag in tags:
        content = re.sub(rf"<{tag}.*?>.*?</{tag}>", "", content, flags=re.DOTALL)
    return re.sub(r"\n{3,}", "\n\n", content).strip()


async def main():
    config = await Config.load("./files/config.yml")
    client = OpenRouterClient(config)

    results = await process_prompts(client, config)

    output_path = Path("./files/generations.yml")

    with open(output_path, "w") as f:
        yaml.dump(results, f)

    print(f"\nTotal cost: ${client.total_cost:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
