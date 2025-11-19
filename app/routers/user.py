from webbrowser import get
from fastapi import APIRouter
from app.services.user import get_user

router = APIRouter(
    prefix="/user",
    tags=["user"],
)

@router.get("/")
def read_user():
    return get_user()
