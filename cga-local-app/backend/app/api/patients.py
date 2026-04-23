from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.session import get_db
from app.models.entities import Patient
from app.schemas.patient import PatientCreate, PatientOut

router = APIRouter(prefix="/api/patients", tags=["patients"])


@router.post("", response_model=PatientOut)
def create_patient(payload: PatientCreate, db: Session = Depends(get_db), _=Depends(get_current_user)) -> Patient:
    patient = Patient(**payload.model_dump())
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return patient


@router.get("", response_model=list[PatientOut])
def list_patients(db: Session = Depends(get_db), _=Depends(get_current_user)) -> list[Patient]:
    return db.query(Patient).order_by(Patient.last_name.asc()).all()
