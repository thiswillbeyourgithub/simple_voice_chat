import json
import requests
from typing import List, Dict, Tuple, Optional, Any
from loguru import logger
import litellm


def get_models_and_costs_from_proxy(
    api_base: str, api_key: Optional[str]
) -> Tuple[List[str], Dict[str, Dict[str, float]]]:
    """Fetches model names and costs from a LiteLLM-compatible proxy's /model/info endpoint."""
    models = []
    costs = {}
    # Construct the correct URL for the /model/info endpoint relative to the api_base
    # Assuming api_base is like "http://host:port/v1"
    if not api_base:
        logger.error("Cannot fetch models from proxy: api_base is not configured.")
        return models, costs
    try:
        # More robustly construct the model info URL
        base_url_parts = api_base.split("/v1")
        if len(base_url_parts) > 0 and base_url_parts[0]:
            model_info_url = f"{base_url_parts[0]}/model/info"
        else:
            # Fallback or alternative logic if '/v1' is not present or structure is different
            # This might need adjustment based on expected api_base formats
            logger.warning(
                f"Could not reliably determine /model/info URL from api_base '{api_base}'. Attempting replacement."
            )
            model_info_url = api_base.replace(
                "/v1", "/model/info"
            )  # Keep original fallback

    except Exception as url_e:
        logger.error(
            f"Error constructing model info URL from api_base '{api_base}': {url_e}"
        )
        return models, costs

    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        logger.info(f"Fetching model info from LLM proxy: {model_info_url}")
        response = requests.get(
            model_info_url, headers=headers, timeout=15
        )  # Added timeout
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        if "data" not in data or not isinstance(data["data"], list):
            logger.error(
                f"Unexpected format in response from {model_info_url}: 'data' key missing or not a list."
            )
            return models, costs

        for item in data["data"]:
            original_model_name = item.get("model_name")
            model_info = item.get("model_info", {})  # Look inside model_info
            input_cost = model_info.get("input_cost_per_token")
            output_cost = model_info.get("output_cost_per_token")

            if original_model_name:
                # Prepend the correct prefix for proxy models
                prefixed_model_name = (
                    f"litellm_proxy/{original_model_name}"  # Corrected prefix
                )
                models.append(prefixed_model_name)  # Add prefixed name to list

                if input_cost is not None and output_cost is not None:
                    # Use the prefixed name as the key in the costs dictionary
                    costs[prefixed_model_name] = {
                        "input_cost_per_token": float(input_cost),
                        "output_cost_per_token": float(output_cost),
                    }
                    logger.debug(
                        f"Loaded cost for {prefixed_model_name} from proxy: Input={input_cost}, Output={output_cost}"
                    )
                else:
                    logger.warning(
                        f"Cost information missing for model '{original_model_name}' (prefixed as '{prefixed_model_name}') in proxy response."
                    )
                    # Use the prefixed name as the key even if costs are missing
                    costs[prefixed_model_name] = {
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    }  # Default to 0 if missing
            else:
                logger.warning("Found item without 'model_name' in proxy response.")

    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP error fetching model info from {model_info_url}: {e}")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON response from {model_info_url}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error fetching model info from proxy: {e}")

    if not models:
        logger.warning(
            f"Could not retrieve any models from the LLM proxy at {model_info_url}."
        )
    return models, costs


# --- Cost Calculation Function (Renamed) ---
def calculate_llm_cost(
    model: str, usage: Dict[str, int], model_cost_data: Dict[str, Dict[str, float]]
) -> Dict[str, Any]:
    """Calculates cost based on LLM usage and provided cost data."""
    # Model name passed here should already be the correct one (prefixed if from proxy)
    cost_data = {
        "input_cost": 0.0,
        "output_cost": 0.0,
        "total_cost": 0.0,
        "model": model,
        "usage": usage,
        "error": None,
    }
    try:
        model_pricing = model_cost_data.get(model)

        if not model_pricing:
            cost_data["error"] = (
                f"Cost data not found for model '{model}' in provided pricing information."
            )
            logger.warning(cost_data["error"])
            return cost_data  # Return with error

        input_cost_per_token = model_pricing.get("input_cost_per_token")
        output_cost_per_token = model_pricing.get("output_cost_per_token")

        if input_cost_per_token is None or output_cost_per_token is None:
            cost_data["error"] = (
                f"Input/Output cost per token missing for model '{model}' in provided pricing information."
            )
            logger.warning(cost_data["error"])
            return cost_data  # Return with error

        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        # Costs are usually per Million tokens, adjust calculation if needed
        # Assuming the costs fetched are already per single token here
        # If they were per Million, divide by 1,000,000
        # Example: input_cost = (prompt_tokens / 1_000_000) * input_cost_per_million_token

        input_cost = prompt_tokens * input_cost_per_token
        output_cost = completion_tokens * output_cost_per_token
        total_cost = input_cost + output_cost

        cost_data.update(
            {
                "input_cost": input_cost,
                "output_cost": output_cost,
                "total_cost": total_cost,
            }
        )
        logger.info(
            f"Calculated cost for {model}: Input=${input_cost:.6f}, Output=${output_cost:.6f}, Total=${total_cost:.6f}"
        )

    except Exception as e:
        error_msg = (
            f"Error calculating cost for model '{model}' using provided data: {e}"
        )
        logger.error(error_msg)
        cost_data["error"] = error_msg

    return cost_data


def get_models_and_costs_from_litellm() -> (
    Tuple[List[str], Dict[str, Dict[str, float]]]
):
    """Fetches model names and costs using litellm.model_cost."""
    models = []
    costs = {}
    try:
        # Ensure LiteLLM has loaded its model costs
        litellm.model_cost  # Accessing this triggers loading if not already done
        available_litellm_models = list(litellm.model_cost.keys())

        if not available_litellm_models:
            logger.warning(
                "LiteLLM reported no available models with cost information (litellm.model_cost)."
            )
            return models, costs

        for model_name in available_litellm_models:
            # No prefix needed for models directly from litellm.model_cost
            model_pricing = litellm.model_cost.get(model_name, {})
            input_cost = model_pricing.get("input_cost_per_token")
            output_cost = model_pricing.get("output_cost_per_token")

            models.append(model_name)
            if input_cost is not None and output_cost is not None:
                costs[model_name] = {
                    "input_cost_per_token": float(input_cost),
                    "output_cost_per_token": float(output_cost),
                }
                logger.debug(
                    f"Loaded cost for {model_name} from litellm.model_cost: Input={input_cost}, Output={output_cost}"
                )
            else:
                logger.warning(
                    f"Cost information missing for model '{model_name}' in litellm.model_cost."
                )
                costs[model_name] = {
                    "input_cost_per_token": 0.0,
                    "output_cost_per_token": 0.0,
                }  # Default to 0

    except Exception as e:
        logger.error(f"Failed to get model list/costs from litellm.model_cost: {e}")

    return models, costs
