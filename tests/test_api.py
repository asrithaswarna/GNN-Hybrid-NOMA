import pytest

fastapi = pytest.importorskip("fastapi")

from api import main as api_main
from api.schemas import RatePredictionRequest


def test_api_health_and_model_info():
    assert api_main.health()["status"] == "ok"
    info = api_main.model_info()
    assert info["task"] == "user rate regression"


def test_api_request_validation():
    request = RatePredictionRequest(x=1.0, y=2.0, distance=10.0, channel_gain=1e-4, min_rate_req_bps=1e5)
    assert request.distance == 10.0
    with pytest.raises(ValueError):
        RatePredictionRequest(x=1.0, y=2.0, distance=0.0, channel_gain=1e-4, min_rate_req_bps=1e5)
