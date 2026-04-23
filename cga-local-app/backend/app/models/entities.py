from datetime import datetime
from enum import Enum

from sqlalchemy import JSON, Boolean, DateTime, Enum as SqlEnum, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


class Role(str, Enum):
    ADMIN = "admin"
    CLINICIAN = "clinician"
    REVIEWER = "reviewer"


class ModuleType(str, Enum):
    CGA = "cga"
    GDS = "gds"
    SARC_F = "sarc_f"
    G8 = "g8"
    CDT = "cdt"


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(150))
    role: Mapped[Role] = mapped_column(SqlEnum(Role), default=Role.CLINICIAN)
    hashed_password: Mapped[str] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    first_name: Mapped[str] = mapped_column(String(120))
    last_name: Mapped[str] = mapped_column(String(120))
    pesel: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    birth_date: Mapped[str | None] = mapped_column(String(32), nullable=True)
    caregiver_name: Mapped[str | None] = mapped_column(String(150), nullable=True)
    caregiver_phone: Mapped[str | None] = mapped_column(String(50), nullable=True)
    oncogeriatric_mode: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    visits: Mapped[list["AssessmentVisit"]] = relationship(back_populates="patient")


class AssessmentVisit(Base):
    __tablename__ = "assessment_visits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"))
    clinician_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    mode: Mapped[str] = mapped_column(String(60), default="interviewer_assisted")
    status: Mapped[str] = mapped_column(String(30), default="in_progress")
    clinician_notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    patient: Mapped[Patient] = relationship(back_populates="visits")
    modules: Mapped[list["AssessmentModuleData"]] = relationship(back_populates="visit")


class AssessmentModuleData(Base):
    __tablename__ = "assessment_module_data"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    visit_id: Mapped[int] = mapped_column(ForeignKey("assessment_visits.id"))
    module_type: Mapped[ModuleType] = mapped_column(SqlEnum(ModuleType))
    payload: Mapped[dict] = mapped_column(JSON)
    score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    interpretation: Mapped[str | None] = mapped_column(String(255), nullable=True)
    scoring_version: Mapped[str] = mapped_column(String(20), default="1.0")
    clinician_override_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    visit: Mapped[AssessmentVisit] = relationship(back_populates="modules")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    actor_username: Mapped[str] = mapped_column(String(100))
    action: Mapped[str] = mapped_column(String(120))
    target_type: Mapped[str] = mapped_column(String(60))
    target_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    details: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
