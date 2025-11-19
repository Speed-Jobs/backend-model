from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import routers_skill_match, user

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
)

app.include_router(user.router)
app.include_router(routers_skill_match.router)