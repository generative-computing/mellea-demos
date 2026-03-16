"""LangFlow Trace Visualization - FastAPI application."""

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.server.routers.trace_visualization.routes import router as trace_router

app = FastAPI(title="LangFlow Trace Visualization")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers (before static mount so API routes take precedence)
app.include_router(trace_router)


@app.get("/healthcheck", tags=["General"])
def healthcheck():
    return {"Project": "LangFlow Trace Visualization", "Status": "OK"}


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"exception": f"{exc!r}"})


# Static files - mounted last so API routes take precedence
app.mount("/", StaticFiles(directory="app/server/static", html=True), name="static")
