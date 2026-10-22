"""Run: python -m evaluation.pairwise.server --study study.json --db votes.sqlite3."""

import argparse
import hmac
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from . import ANNOTATION_PROTOCOL
from .manifest import load_study
from .store import Conflict, Store

STATIC = Path(__file__).parent / "static"
COOKIE = "memorybench_pairwise"


class Body(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Session(Body):
    name: str = Field(default="", max_length=60)
    resume_code: str | None = Field(default=None, max_length=128)


class Answers(Body):
    choice: Literal["A", "B", "tie"] | None = None
    elapsed_seconds: float = Field(default=0, ge=0, le=1e8)


class Save(Body):
    revision: int = Field(ge=0)
    answers: Answers


class Issue(Body):
    description: str = Field(min_length=1, max_length=2000)


def create_app(study_path, db_path, secure_cookie=False, *, open_assignment=False, random_assignment=False):
    study, signature, tasks = load_study(study_path)
    store = Store(db_path, study, signature, tasks, open_assignment=open_assignment,
                  random_assignment=random_assignment)
    cookie = COOKIE + "_" + store.cookie_namespace
    app = FastAPI(title="MemoryBench Pairwise", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.store = store

    @app.middleware("http")
    async def privacy(request, call_next):
        # Same-origin mutation, including login; no cross-site cookie/session writes.
        origin = request.headers.get("origin")
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            expected = f'{request.url.scheme}://{request.headers.get("host", "")}'
            if origin and not hmac.compare_digest(origin, expected):
                return JSONResponse({"detail": "Cross-origin request rejected"}, status_code=403)
            if request.headers.get("sec-fetch-site") == "cross-site":
                return JSONResponse({"detail": "Cross-site request rejected"}, status_code=403)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; media-src 'self'; img-src 'self' data:; frame-ancestors 'none'; object-src 'none'"
        return response

    @app.exception_handler(Conflict)
    async def conflict(_request, error):
        return JSONResponse({"detail": str(error)}, status_code=409)

    @app.exception_handler(ValueError)
    async def invalid(_request, error):
        return JSONResponse({"detail": str(error)}, status_code=400)

    @app.exception_handler(KeyError)
    async def missing(_request, _error):
        return JSONResponse({"detail": "Not found"}, status_code=404)

    def user(request: Request):
        result = store.session(request.cookies.get(cookie))
        if result is None:
            raise HTTPException(401, "请先登记评估者代号或输入续评码")
        return result

    @app.get("/api/config")
    def config():
        return {"title": study.title, "raters_per_pair": study.raters_per_pair,
                "open_assignment": store.open_assignment,
                "random_assignment": store.random_assignment,
                "models": len(study.models), "cases": len(study.cases), "total_tasks": len(tasks),
                "session_id": store.session_namespace,
                "study_id": signature[:16], "annotation_protocol": ANNOTATION_PROTOCOL}

    @app.post("/api/session")
    def login(body: Session, response: Response):
        person, token = store.register(body.name, body.resume_code)
        response.set_cookie(cookie, token, httponly=True, samesite="strict", secure=secure_cookie,
                            max_age=365 * 86400)
        return {"name": person["name"], "resume_code": token}

    @app.get("/api/session")
    def current(person=Depends(user)):
        return {"name": person["name"]}

    @app.post("/api/logout")
    def logout(response: Response):
        response.delete_cookie(cookie)
        return {"ok": True}

    @app.get("/api/next")
    def next_task(person=Depends(user)):
        task, progress = store.next(person["id"])
        return {"task": task, "progress": progress}

    @app.post("/api/task/{aid}/heartbeat")
    def heartbeat(aid: str, person=Depends(user)):
        store.heartbeat(person["id"], aid)
        return {"ok": True}

    @app.post("/api/task/{aid}/draft")
    def draft(aid: str, body: Save, person=Depends(user)):
        return {"revision": store.save(person["id"], aid, body.revision, body.answers.model_dump())}

    @app.post("/api/task/{aid}/vote")
    def vote(aid: str, body: Save, person=Depends(user)):
        return {"revision": store.save(person["id"], aid, body.revision, body.answers.model_dump(), submit=True)}

    @app.post("/api/task/{aid}/issue")
    def issue(aid: str, body: Issue, person=Depends(user)):
        store.issue(person["id"], aid, body.description)
        return {"ok": True}

    @app.api_route("/api/media/{aid}/{slot}", methods=["GET", "HEAD"])
    def media(aid: str, slot: str, person=Depends(user)):
        path = store.media(person["id"], aid, slot)
        # Opaque URLs and a neutral filename; Range is handled by Starlette.
        return FileResponse(path, media_type="video/mp4", filename="video.mp4", content_disposition_type="inline")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8920)
    parser.add_argument("--secure-cookie", action="store_true", help="Use when serving behind HTTPS")
    parser.add_argument("--open-assignment", action="store_true",
                        help="Allow every reviewer to judge every task once, beyond the per-pair target")
    parser.add_argument("--random-assignment", action="store_true",
                        help="Sample tasks uniformly with replacement; no reviewer or per-pair quota")
    args = parser.parse_args()
    app = create_app(args.study, args.db, args.secure_cookie,
                     open_assignment=args.open_assignment, random_assignment=args.random_assignment)
    uvicorn.run(app, host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
