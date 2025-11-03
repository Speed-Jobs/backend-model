from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import user

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
)

app.include_router(user.router)