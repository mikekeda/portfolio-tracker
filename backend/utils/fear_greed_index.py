import logging

from typing import Optional, Union

import aiohttp


async def gen_fear_greed_index() -> Optional[dict[str, Union[str, float]]]:
    """Scrape Fear & Greed Index from CNN"""
    url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"

    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

    try:
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(verify_ssl=False), raise_for_status=True
        ) as session:
            async with session.get(url, headers=headers) as response:
                data = (await response.json())["fear_and_greed"]

                # Extract the index value
                fear_greed_value = data["score"]
                fear_greed_label = data["rating"]  # "Extreme Fear", "Greed", etc.

                return {"value": fear_greed_value, "label": fear_greed_label, "timestamp": data["timestamp"]}
    except Exception as e:
        logging.warning(f"Error fetching Fear & Greed Index: {e}")
        return None
