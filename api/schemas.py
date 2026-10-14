from pydantic import BaseModel, Field


class RatePredictionRequest(BaseModel):
    x: float
    y: float
    distance: float = Field(gt=0)
    channel_gain: float = Field(ge=0)
    min_rate_req_bps: float = Field(ge=0)


class RatePredictionResponse(BaseModel):
    predicted_rate_mbps: float
    model: str
