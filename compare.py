import asyncio
import random
from pathlib import Path
from typing import Dict
from src.api_client import OpenRouterClient
from src.config import Config
from src.yaml_config import yaml


def interleave_generations(generations: Dict) -> str:
    prompt_types = list(generations.keys())
    assert len(prompt_types) == 2

    gen1, gen2 = list(generations[prompt_types[0]]), list(generations[prompt_types[1]])
    assert len(gen1) == len(gen2)

    random.shuffle(gen1)
    random.shuffle(gen2)

    interleaved = []
    for g1, g2 in zip(gen1, gen2):
        interleaved.extend(
            [
                f"# Prompt Variation #1\n\n{g1}\n\n---",
                f"# Prompt Variation #2\n\n{g2}\n\n---",
            ]
        )

    return "\n\n".join(interleaved)


async def compare(client: OpenRouterClient, config: Config, generations: Dict) -> str:
    interleaved_content = interleave_generations(generations)

    response = await client.request_chat_completion(
        {
            "model": config.model,
            "messages": [
                {"role": "system", "content": config.comparison_prompt},
                {"role": "user", "content": interleaved_content},
            ],
        }
    )

    return response


async def main():
    config = await Config.load("./files/config.yml")
    client = OpenRouterClient(config)

    generations_path = Path("./files/generations.yml")
    with open(generations_path) as f:
        generations = yaml.load(f)

    analysis = await compare(client, config, generations)

    print(analysis)
    print(f"\nTotal cost: ${client.total_cost:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
