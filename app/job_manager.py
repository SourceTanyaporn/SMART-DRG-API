from uuid import uuid4
from typing import Any


jobs: dict[str, dict[str, Any]] = {}


def create_job() -> str:
    job_id = str(uuid4())

    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "message": "รอประมวลผล",
        "result": None,
        "error": None,
    }

    return job_id


def update_job(
    job_id: str,
    *,
    status: str,
    progress: int,
    message: str,
    result: Any = None,
    error: str | None = None,
):
    if job_id not in jobs:
        return

    jobs[job_id].update({
        "status": status,
        "progress": progress,
        "message": message,
        "result": result,
        "error": error,
    })


def get_job(job_id: str):
    return jobs.get(job_id)