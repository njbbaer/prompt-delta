import asyncio
import re
from typing import Dict
from pathlib import Path
from tqdm import tqdm
from typing import List
from ruamel.yaml.scalarstring import LiteralScalarString

from src.api_client import OpenRouterClient
from src.config import Config
from src.yaml_config import yaml


async def generate_batch(
    client: OpenRouterClient, config: Config, prompt: str, pbar: tqdm
) -> List[str]:
    async def make_request():
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
                                "cache_control": {
                                    "type": "ephemeral",
                                },
                            }
                        ],
                    },
                    {"role": "user", "content": config.content_message},
                ],
            }
        )
        pbar.update(1)
        response = strip_tags(response, config.strip_tags)
        return LiteralScalarString(response)

    tasks = [make_request() for _ in range(config.batch_size)]
    return await asyncio.gather(*tasks)


async def process_prompts(
    client: OpenRouterClient,
    config: Config,
) -> Dict:
    tasks = []
    with tqdm(total=config.total_calls, desc="Processing") as pbar:
        for _name, prompt in config.content_prompts.items():
            tasks.append(generate_batch(client, config, prompt, pbar))

        results = await asyncio.gather(*tasks)
        return dict(zip(config.content_prompts.keys(), results))


def strip_tags(content: str, tags: List[str]) -> str:
    for tag in tags:
        content = re.sub(rf"<{tag}.*?>.*?</{tag}>", "", content, flags=re.DOTALL)
    content = re.sub(r"\n{3,}", "\n\n", content)
    return content.strip()


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
