from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.exceptions import RequestValidationError
from routers.chat import router as chat_router

app = FastAPI(
    title="AI Support Assistant",
    description="LLM-powered support assistant using OpenRouter (Llama 3.1 405B).",
    version="1.0.0",
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Override FastAPI's default 422 handler to return a cleaner error shape.

    Args:
        request: The incoming HTTP request.
        exc: The validation error raised by Pydantic.

    Returns:
        JSONResponse with status 422 and a list of field-level error details.
    """
    errors = [
        {"field": ".".join(str(loc) for loc in e["loc"]), "message": e["msg"]}
        for e in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": errors})


# Register routers
app.include_router(chat_router)


@app.get("/")
async def index():
    return FileResponse("index.html")
