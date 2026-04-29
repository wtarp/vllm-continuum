import logging
import os
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any

from openai import BadRequestError, OpenAI
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from minisweagent.models import GLOBAL_MODEL_STATS

logger = logging.getLogger("sglang_model")


class ContextLengthExceededError(Exception):
    """Raised when the context length exceeds the model's maximum."""
    pass


@dataclass
class SglangModelConfig:
    model_name: str = "meta-llama/Llama-3.1-70B-Instruct"
    base_url: str = "http://localhost:30000/v1"
    port: int | None = None  # Override port in base_url if specified
    api_key: str = "EMPTY"  # SGLang doesn't require API key
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    job_id: int = 0  # Instance/job ID for job-aware routing
    routing_key: str | None = None  # Explicit routing key override
    session_params: dict[str, Any] = field(default_factory=dict)  # Optional SGLang session metadata
    stream: bool = True  # Enable streaming responses
    timeout: float = 900.0  # Request timeout in seconds (default: 900s)
    max_completion_tokens: int = 2048  # Maximum tokens to generate per response


class SglangModel:
    """Model class for SGLang inference server.

    This class provides an interface to SGLang (https://github.com/sgl-project/sglang),
    a fast serving framework for large language models and vision language models.

    SGLang exposes an OpenAI-compatible API, so we use the OpenAI client to communicate with it.

    Args:
        config_class: Configuration class to use (default: SglangModelConfig)
        **kwargs: Additional arguments to pass to the config class

    Example:
        >>> model = SglangModel(base_url="http://localhost:30000/v1", model_name="meta-llama/Llama-3.1-70B-Instruct")
        >>> response = model.query([{"role": "user", "content": "Hello!"}])
    """

    def __init__(self, *, config_class: type = SglangModelConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.cost = 0.0  # SGLang doesn't have cost, set to 0
        self.n_calls = 0

        # Override port in base_url if port is specified
        base_url = self.config.base_url
        if self.config.port is not None:
            # Parse and replace port in base_url
            import re
            base_url = re.sub(r':\d+/', f':{self.config.port}/', base_url)
            if not re.search(r':\d+/', base_url):
                # If no port found in URL, add it before the path
                base_url = re.sub(r'(https?://[^/]+)', rf'\1:{self.config.port}', base_url)

        # Initialize OpenAI client pointing to SGLang server
        self.client = OpenAI(
            base_url=base_url,
            api_key=self.config.api_key,
            timeout=self.config.timeout,
        )

    @staticmethod
    def _filter_openai_params(params: dict[str, Any]) -> dict[str, Any]:
        """Filter parameters to only include those compatible with OpenAI API."""
        # List of valid parameters for OpenAI chat completions API
        # Based on: https://platform.openai.com/docs/api-reference/chat/create
        valid_params = {
            "temperature",
            "top_p",
            "n",
            "stream",
            "stop",
            "max_tokens",
            "max_completion_tokens",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "user",
            "seed",
            "tools",
            "tool_choice",
            "response_format",
            "logprobs",
            "top_logprobs",
        }
        filtered = {k: v for k, v in params.items() if k in valid_params}
        if filtered != params:
            dropped = set(params.keys()) - set(filtered.keys())
            logger.debug(f"Dropped incompatible parameters for OpenAI API: {dropped}")
        return filtered

    @classmethod
    def _split_request_params(
        cls, params: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Split OpenAI-compatible params from SGLang-specific request metadata."""
        params = dict(params)
        extra_body = params.pop("extra_body", None)
        if extra_body is None:
            merged_extra_body: dict[str, Any] = {}
        elif isinstance(extra_body, dict):
            merged_extra_body = dict(extra_body)
        else:
            raise TypeError("extra_body must be a dict when provided")

        filtered_params = cls._filter_openai_params(params)
        for key, value in params.items():
            if key not in filtered_params:
                merged_extra_body[key] = value

        return filtered_params, merged_extra_body

    def _build_extra_body(
        self, params: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        filtered_params, extra_body = self._split_request_params(params)
        if self.config.session_params:
            extra_body.setdefault("session_params", deepcopy(self.config.session_params))
        if self.config.job_id > 0:
            extra_body.setdefault(
                "routing_key",
                self.config.routing_key or str(self.config.job_id),
            )
        elif self.config.routing_key is not None:
            extra_body.setdefault("routing_key", self.config.routing_key)
        return filtered_params, extra_body

    @retry(
        stop=stop_after_attempt(int(os.getenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "10"))),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type(
            (
                KeyboardInterrupt,
                ContextLengthExceededError,
            )
        ),
    )
    def _query(self, messages: list[dict[str, str]], **kwargs):
        """Internal query method with retry logic."""
        try:
            # Merge config model_kwargs and runtime kwargs, then filter for OpenAI compatibility
            all_params = self.config.model_kwargs | kwargs

            # Add stream parameter from config if not explicitly provided
            if "stream" not in all_params:
                all_params["stream"] = self.config.stream

            # Add max_completion_tokens from config if not explicitly provided
            if "max_completion_tokens" not in all_params and "max_tokens" not in all_params:
                all_params["max_completion_tokens"] = self.config.max_completion_tokens

            filtered_params, extra_body = self._build_extra_body(all_params)

            request_kwargs: dict[str, Any] = {
                "model": self.config.model_name,
                "messages": messages,
                **filtered_params,
            }
            if extra_body:
                request_kwargs["extra_body"] = extra_body

            return self.client.chat.completions.create(**request_kwargs)
        except BadRequestError as e:
            # Check if this is a context length exceeded error
            error_message = str(e)
            if "maximum context length" in error_message.lower() or "context length" in error_message.lower():
                logger.error(f"Context length exceeded: {error_message}")
                raise ContextLengthExceededError(error_message) from e
            logger.error(f"Bad request error querying SGLang server: {e}")
            raise
        except Exception as e:
            logger.error(f"Error querying SGLang server: {e}")
            raise

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        """Query the SGLang model with a list of messages.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional arguments to pass to the completion API

        Returns:
            Dictionary with 'content' and 'extra' keys
        """
        response = self._query(messages, **kwargs)

        # Track calls but no cost for SGLang
        self.n_calls += 1
        cost = 0.0  # No cost for SGLang
        self.cost += cost
        GLOBAL_MODEL_STATS.add(cost)

        # Check if streaming is enabled
        use_stream = kwargs.get("stream", self.config.stream)

        if use_stream:
            # Handle streaming response
            content_chunks = []
            full_response = None

            for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content:
                        content_chunks.append(delta.content)
                # Keep the last chunk for metadata
                full_response = chunk

            content = "".join(content_chunks)

            return {
                "content": content,
                "extra": {
                    "response": full_response.model_dump() if full_response else {},
                    "streamed": True,
                },
            }
        else:
            # Handle non-streaming response
            return {
                "content": response.choices[0].message.content or "",
                "extra": {
                    "response": response.model_dump(),
                    "streamed": False,
                },
            }

    def get_template_vars(self) -> dict[str, Any]:
        """Get template variables for this model instance."""
        return asdict(self.config) | {"n_model_calls": self.n_calls, "model_cost": self.cost}
