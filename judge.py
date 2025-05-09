import asyncio
import random
from pathlib import Path
from typing import Dict
from tqdm import tqdm
from ruamel.yaml.scalarstring import LiteralScalarString

from src.api_client import OpenRouterClient
from src.config import Config
from src.yaml_config import yaml


def interleave_generations(config: Config, generations: Dict) -> str:
    interleaved = []
    for i, (content_name, content_gens) in enumerate(generations.items(), start=1):
        content_variation_source = config.content_variations[content_name]
        interleaved.append(f'## Variation "{i}: {content_variation_source}"')

        authors = list(content_gens.keys())
        author_generations = []
        sample_lengths = set()

        for author in authors:
            gen = list(content_gens[author])
            sample_lengths.add(len(gen))
            random.shuffle(gen)
            author_generations.append(gen)

        assert (
            len(sample_lengths) == 1
        ), "All prompts must have the same number of samples"

        for author_idx, author_gen in enumerate(author_generations, start=1):
            for j, content in enumerate(author_gen, start=1):
                interleaved.append(
                    f"### Variation {i}, Author {author_idx}, Sample {j}"
                )
                interleaved.append(content)

    return "\n\n".join(interleaved)


async def judge(client: OpenRouterClient, config: Config, generations: Dict) -> str:
    interleaved_content = interleave_generations(config, generations)

    response = await client.request_chat_completion(
        {
            "model": config.judge_model,
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

    judge_tasks = []
    for _ in range(config.num_judges):
        judge_task = judge(client, config, generations)
        judge_tasks.append(judge_task)

    judgements = []
    with tqdm(total=len(judge_tasks), desc="Judging") as pbar:
        for future in asyncio.as_completed(judge_tasks):
            judgment = await future
            judgements.append(LiteralScalarString(judgment))
            pbar.update(1)

    with open("./files/judgements.yml", "w") as f:
        yaml.dump(judgements, f)

    print(f"\nTotal cost: ${client.total_cost:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
