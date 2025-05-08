import asyncio
import random
from pathlib import Path
from typing import Dict
from src.api_client import OpenRouterClient
from src.config import Config
from src.yaml_config import yaml


def interleave_generations(config: Config, generations: Dict) -> str:
    interleaved = []
    for i, (content_name, content_gens) in enumerate(generations.items(), start=1):
        content_variation_source = config.content_variations[content_name]
        interleaved.append(f'## Variation "{i}: {content_variation_source}"')

        prompt_types = list(content_gens.keys())
        assert len(prompt_types) == 3

        gen1, gen2, gen3 = (  # Added gen3
            list(content_gens[prompt_types[0]]),
            list(content_gens[prompt_types[1]]),
            list(content_gens[prompt_types[2]]),
        )
        assert len(gen1) == len(gen2) == len(gen3)

        random.shuffle(gen1)
        random.shuffle(gen2)
        random.shuffle(gen3)

        for j, content in enumerate(gen1, start=1):
            interleaved.append(f"### Variation {i}, Author 1, Sample {j}")
            interleaved.append(content)

        for j, content in enumerate(gen2, start=1):
            interleaved.append(f"### Variation {i}, Author 2, Sample {j}")
            interleaved.append(content)

        for j, content in enumerate(gen3, start=1):  # Added loop for gen3
            interleaved.append(f"### Variation {i}, Author 3, Sample {j}")
            interleaved.append(content)

    return "\n\n".join(interleaved)


async def judge(client: OpenRouterClient, config: Config, generations: Dict) -> str:
    interleaved_content = interleave_generations(config, generations)

    response = await client.request_chat_completion(
        {
            "model": config.model,
            "messages": [
                {"role": "system", "content": config.judge_prompt},
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

    analysis = await judge(client, config, generations)

    print(analysis)
    print(f"\nTotal cost: ${client.total_cost:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
