from datetime import datetime

from pydantic import BaseModel, Field

from app.models.entities import ModuleType


class VisitCreate(BaseModel):
    patient_id: int
    mode: str = "interviewer_assisted"


class VisitOut(BaseModel):
    id: int
    patient_id: int
    clinician_id: int
    started_at: datetime
    completed_at: datetime | None
    mode: str
    status: str

    model_config = {"from_attributes": True}


class ModuleSubmit(BaseModel):
    module_type: ModuleType
    payload: dict = Field(default_factory=dict)
    clinician_override_reason: str | None = None


class ModuleOut(BaseModel):
    id: int
    module_type: ModuleType
    payload: dict
    score: int | None
    interpretation: str | None
    scoring_version: str

    model_config = {"from_attributes": True}
