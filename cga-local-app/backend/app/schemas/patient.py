from datetime import datetime

from pydantic import BaseModel


class PatientCreate(BaseModel):
    first_name: str
    last_name: str
    pesel: str | None = None
    birth_date: str | None = None
    caregiver_name: str | None = None
    caregiver_phone: str | None = None
    oncogeriatric_mode: bool = False


class PatientOut(PatientCreate):
    id: int
    created_at: datetime

    model_config = {"from_attributes": True}
