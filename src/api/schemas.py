"""
schemas.py
==========
Pydantic request/response models for the REST API. These double as the
OpenAPI schema shown at ``/docs`` and ``/redoc``.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
class IncidentRequest(BaseModel):
    """Minimal description of a traffic incident to score."""

    event_cause: str = Field(..., examples=["accident"],
                             description="Cause of the incident (e.g. accident, pot_holes).")
    corridor: str = Field("Non-corridor", examples=["Mysore Road"])
    zone: str = Field("unknown", examples=["Central Zone 2"])
    veh_type: str = Field("not_applicable", examples=["heavy_vehicle"])
    event_type: str = Field("unplanned", examples=["unplanned"])
    status: str = Field("active", examples=["active"])
    hour: int = Field(9, ge=0, le=23, description="Local hour of day (0-23).")
    latitude: float = Field(12.9716, ge=12.6, le=13.4)
    longitude: float = Field(77.5946, ge=77.2, le=77.9)

    model_config = {
        "json_schema_extra": {
            "example": {
                "event_cause": "accident", "corridor": "Hosur Road",
                "zone": "South Zone 2", "veh_type": "heavy_vehicle",
                "event_type": "unplanned", "status": "active",
                "hour": 18, "latitude": 12.9081, "longitude": 77.6476,
            }
        }
    }


class TaskPrediction(BaseModel):
    label: Optional[str] = None
    probability: Optional[float] = None
    value: Optional[float] = None
    unit: Optional[str] = None


class PredictionResponse(BaseModel):
    priority: Optional[TaskPrediction] = None
    closure: Optional[TaskPrediction] = None
    resolution: Optional[TaskPrediction] = Field(
        default=None, description="Predicted time-to-resolution in hours (real target).")
    model_versions: dict = Field(default_factory=dict)


# --------------------------------------------------------------------------
# Health / meta
# --------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: list[str]
    artifacts_available: dict[str, bool]


class MessageResponse(BaseModel):
    detail: str
