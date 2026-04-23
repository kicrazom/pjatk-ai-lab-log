from pydantic import BaseModel

from app.models.entities import Role


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    id: int
    username: str
    full_name: str
    role: Role

    model_config = {"from_attributes": True}
