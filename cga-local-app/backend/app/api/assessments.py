from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.session import get_db
from app.models.entities import AssessmentModuleData, AssessmentVisit, ModuleType, User
from app.schemas.assessment import ModuleOut, ModuleSubmit, VisitCreate, VisitOut
from app.services.pdf import build_summary_pdf
from app.services.scoring import calculate_g8, calculate_gds, calculate_sarc_f

router = APIRouter(prefix="/api/assessments", tags=["assessments"])


@router.post("/visits", response_model=VisitOut)
def create_visit(payload: VisitCreate, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> AssessmentVisit:
    visit = AssessmentVisit(patient_id=payload.patient_id, clinician_id=user.id, mode=payload.mode)
    db.add(visit)
    db.commit()
    db.refresh(visit)
    return visit


@router.post("/visits/{visit_id}/modules", response_model=ModuleOut)
def submit_module(visit_id: int, payload: ModuleSubmit, db: Session = Depends(get_db), _=Depends(get_current_user)) -> AssessmentModuleData:
    visit = db.query(AssessmentVisit).filter(AssessmentVisit.id == visit_id).first()
    if not visit:
        raise HTTPException(status_code=404, detail="Nie znaleziono wizyty")

    score = None
    interpretation = None
    scoring_version = "manual-1.0"
    if payload.module_type == ModuleType.GDS:
        result = calculate_gds(payload.payload, payload.payload.get("variant", "gds15"))
        score, interpretation, scoring_version = result.score, result.interpretation, result.scoring_version
    elif payload.module_type == ModuleType.SARC_F:
        result = calculate_sarc_f(payload.payload)
        score, interpretation, scoring_version = result.score, result.interpretation, result.scoring_version
    elif payload.module_type == ModuleType.G8:
        result = calculate_g8(payload.payload)
        score, interpretation, scoring_version = result.score, result.interpretation, result.scoring_version

    module = AssessmentModuleData(
        visit_id=visit_id,
        module_type=payload.module_type,
        payload=payload.payload,
        score=score,
        interpretation=interpretation,
        scoring_version=scoring_version,
        clinician_override_reason=payload.clinician_override_reason,
    )
    db.add(module)
    db.commit()
    db.refresh(module)
    return module


@router.get("/visits/{visit_id}/summary")
def get_summary(visit_id: int, db: Session = Depends(get_db), _=Depends(get_current_user)) -> dict:
    visit = db.query(AssessmentVisit).filter(AssessmentVisit.id == visit_id).first()
    if not visit:
        raise HTTPException(status_code=404, detail="Nie znaleziono wizyty")

    modules = db.query(AssessmentModuleData).filter(AssessmentModuleData.visit_id == visit_id).all()
    return {
        "visit_id": visit.id,
        "patient_id": visit.patient_id,
        "status": visit.status,
        "scores": [
            {
                "module": m.module_type.value,
                "score": m.score,
                "interpretation": m.interpretation,
                "version": m.scoring_version,
            }
            for m in modules
        ],
    }


@router.get("/visits/{visit_id}/summary.pdf")
def export_summary_pdf(visit_id: int, db: Session = Depends(get_db), _=Depends(get_current_user)) -> Response:
    summary = get_summary(visit_id, db)
    pdf = build_summary_pdf(summary)
    return Response(content=pdf, media_type="application/pdf")
