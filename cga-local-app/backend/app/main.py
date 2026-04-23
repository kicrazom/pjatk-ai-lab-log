from fastapi import FastAPI

from app.api import assessments, auth, patients
from app.core.security import get_password_hash
from app.db.session import Base, SessionLocal, engine
from app.models.entities import Role, User

app = FastAPI(title="CGA Local API", version="0.1.0")
app.include_router(auth.router)
app.include_router(patients.router)
app.include_router(assessments.router)


@app.on_event("startup")
def startup() -> None:
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    if not db.query(User).filter(User.username == "admin").first():
        db.add(
            User(
                username="admin",
                full_name="Administrator lokalny",
                role=Role.ADMIN,
                hashed_password=get_password_hash("admin123"),
            )
        )
        db.commit()
    db.close()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
