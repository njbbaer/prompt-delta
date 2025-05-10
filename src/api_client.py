import httpx
import asyncio

from src.config import Config
from src.logger import Logger


class OpenRouterClient:
    def __init__(self, config: Config):
        self.logger = Logger(config.log_file)
        self.config = config
        self.total_cost = 0.0
        self.cache_discount = 0.0

    async def request_completion(self, params: dict, validator=None):
        for attempt in range(self.config.max_retries):
            try:
                response = await self._make_request(params)

                if not response:
                    raise Exception("Empty response")

                if validator and not await validator(response):
                    raise Exception("Validation failed")

                return response

            except Exception as e:
                print(f"Error on attempt {attempt + 1}: {str(e)}")
                if attempt == self.config.max_retries - 1:
                    raise

        raise Exception("Failed to get a valid response after max retries")

    async def _make_request(self, params: dict):
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=params,
            )
            response.raise_for_status()
            body = response.json()

            if "error" in body:
                raise Exception(body["error"])

            content = body["choices"][0]["message"]["content"]
            details = await self._fetch_details(body["id"])

            cost = details["data"]["total_cost"]
            cache_discount = details["data"].get("cache_discount") or 0.0

            self.total_cost += cost
            self.cache_discount += cache_discount

            self.logger.log(body["id"], cost, cache_discount, params, content)
            return content

    async def _fetch_details(self, generation_id: str):
        details_url = f"https://openrouter.ai/api/v1/generation?id={generation_id}"

        for _ in range(10):
            try:
                async with httpx.AsyncClient(timeout=3) as client:
                    response = await client.get(
                        details_url,
                        headers={"Authorization": f"Bearer {self.config.api_key}"},
                    )
                    response.raise_for_status()
                    return response.json()
            except httpx.HTTPError:
                await asyncio.sleep(0.5)

        raise Exception("Details request timed out")
