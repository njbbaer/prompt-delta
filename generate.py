import asyncio
import re
import sys
from typing import Dict, List, Tuple, Coroutine, Any
from pathlib import Path
from tqdm import tqdm
from ruamel.yaml.scalarstring import LiteralScalarString

from src.api_client import OpenRouterClient
from src.config import GenerationConfig
from src.yaml_config import yaml


async def make_request(
    client: OpenRouterClient,
    prompt: str,
    content_message: str,
    config: GenerationConfig,
    pbar: tqdm,
    semaphore: asyncio.Semaphore,
) -> str:
    async def validator(response: str) -> bool:
        return all(
            f"<{tag}" in response and f"</{tag}>" in response
            for tag in config.response_tags
        )

    async with semaphore:
        response = await client.request_completion(
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
                    {"role": "user", "content": content_message},
                ],
            },
            validator=validator,
        )
        pbar.update(1)
        return response_tags(response, config.response_tags)


async def execute_batch(
    tasks: List[Tuple[str, str, Coroutine[Any, Any, str]]],
) -> Dict[str, Dict[str, List[str]]]:
    results: Dict[str, Dict[str, List[str]]] = {}
    completed = await asyncio.gather(*[task for _, _, task in tasks])

    for (content_name, prompt_name, _), result in zip(tasks, completed):
        if content_name not in results:
            results[content_name] = {}
        if prompt_name not in results[content_name]:
            results[content_name][prompt_name] = []
        results[content_name][prompt_name].append(LiteralScalarString(result))

    return results


async def process_prompts(client: OpenRouterClient, config: GenerationConfig) -> Dict:
    total_generations = (
        len(config.content_prompts)
        * len(config.content_variations)
        * config.iterations
    )

    semaphore = asyncio.Semaphore(20)

    with tqdm(total=total_generations, desc="Generating") as pbar:
        results: Dict[str, Dict[str, List[str]]] = {}

        if config.warm_cache:
            warm_tasks = [
                (
                    content_name,
                    prompt_name,
                    make_request(client, prompt, message, config, pbar, semaphore),
                )
                for prompt_name, prompt in config.content_prompts.items()
                for content_name, message in config.content_variations.items()
                if list(config.content_variations.keys()).index(content_name) == 0
            ]
            warm_results = await execute_batch(warm_tasks)
            results.update(warm_results)

        remaining_size = config.iterations - (1 if config.warm_cache else 0)

        remaining_tasks = [
            (
                content_name,
                prompt_name,
                make_request(client, prompt, message, config, pbar, semaphore),
            )
            for prompt_name, prompt in config.content_prompts.items()
            for content_name, message in config.content_variations.items()
            for _ in range(
                remaining_size
                if (
                    not config.warm_cache
                    or list(config.content_variations.keys()).index(content_name) == 0
                )
                else config.iterations
            )
        ]

        remaining_results = await execute_batch(remaining_tasks)

        for content_name, prompt_dict in remaining_results.items():
            if content_name not in results:
                results[content_name] = {}
            for prompt_name, generations in prompt_dict.items():
                if prompt_name not in results[content_name]:
                    results[content_name][prompt_name] = []
                results[content_name][prompt_name].extend(generations)

    return results


def response_tags(content: str, tags: List[str]) -> str:
    for tag in tags:
        content = re.sub(rf"<{tag}.*?>.*?</{tag}>", "", content, flags=re.DOTALL)
    return re.sub(r"\n{3,}", "\n\n", content).strip()


async def main():
    config_path = sys.argv[1]

    config = GenerationConfig.load(config_path)
    client = OpenRouterClient(config)

    results = await process_prompts(client, config)

    output_file = Path(config.output_file)
    with open(output_file, "w") as f:
        yaml.dump(results, f)

    print()
    print(f"Cache discount: ${client.cache_discount:.2f}")
    print(f"Total cost: ${client.total_cost:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
