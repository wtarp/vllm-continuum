from unittest.mock import MagicMock, patch

from minisweagent.models.sglang_model import SglangModel, SglangModelConfig


def test_sglang_model_config():
    """Test SglangModelConfig creation."""
    config = SglangModelConfig(
        model_name="sglang-model",
        job_id=7,
        routing_key="job-7",
        session_params={"id": "session-7"},
        model_kwargs={"temperature": 0.1},
    )
    assert config.model_name == "sglang-model"
    assert config.job_id == 7
    assert config.routing_key == "job-7"
    assert config.session_params == {"id": "session-7"}
    assert config.model_kwargs == {"temperature": 0.1}


def test_sglang_model_query_forwards_routing_and_session_metadata():
    """Test SGLang request construction for continuum-style metadata."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "ok"
    mock_response.choices = [MagicMock(message=mock_message)]
    mock_response.model_dump.return_value = {"ok": True}
    mock_client.chat.completions.create.return_value = mock_response

    with patch("minisweagent.models.sglang_model.OpenAI", return_value=mock_client):
        model = SglangModel(
            model_name="sglang-model",
            job_id=42,
            session_params={"id": "session-42"},
            model_kwargs={
                "temperature": 0.2,
                "priority": 5,
            },
        )

        messages = [{"role": "user", "content": "Hello"}]
        result = model.query(messages, stream=False)

    assert result["content"] == "ok"
    mock_client.chat.completions.create.assert_called_once_with(
        model="sglang-model",
        messages=messages,
        temperature=0.2,
        stream=False,
        max_completion_tokens=2048,
        extra_body={
            "priority": 5,
            "session_params": {"id": "session-42"},
            "routing_key": "42",
        },
    )


def test_sglang_model_runtime_extra_body_overrides_defaults():
    """Test runtime SGLang metadata can override config defaults."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "done"
    mock_response.choices = [MagicMock(message=mock_message)]
    mock_response.model_dump.return_value = {"done": True}
    mock_client.chat.completions.create.return_value = mock_response

    with patch("minisweagent.models.sglang_model.OpenAI", return_value=mock_client):
        model = SglangModel(
            model_name="sglang-model",
            job_id=9,
            routing_key="default-job",
            session_params={"id": "default-session"},
        )

        model.query(
            [{"role": "user", "content": "Hi"}],
            stream=False,
            extra_body={
                "routing_key": "override-job",
                "session_params": {"id": "override-session"},
                "extra_key": "tenant-a",
            },
        )

    mock_client.chat.completions.create.assert_called_once_with(
        model="sglang-model",
        messages=[{"role": "user", "content": "Hi"}],
        stream=False,
        max_completion_tokens=2048,
        extra_body={
            "routing_key": "override-job",
            "session_params": {"id": "override-session"},
            "extra_key": "tenant-a",
        },
    )
