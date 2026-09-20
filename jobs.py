"""Job latar belakang sederhana: kumpulan thread + status yang bisa di-poll klien."""
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List

import config


class Job:
    def __init__(self, kind: str, stages: List[str]):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.stages = stages
        self.stage = 0
        self.progress = 0.0
        self.status = "queued"
        self.result = None
        self.error = None
        self.created = time.time()
        self.times: List[float | None] = [None] * len(stages)
        self._stage_t0 = time.time()
        self._lock = threading.Lock()

    def set(self, stage: int, progress: float = 0.0) -> None:
        """Pindah ke tahap `stage` atau perbarui progres di dalam tahap itu (0..1)."""
        with self._lock:
            if stage != self.stage:
                now = time.time()
                self.times[self.stage] = now - self._stage_t0
                self._stage_t0 = now
                self.stage = stage
                self.progress = 0.0
            self.progress = max(self.progress, min(1.0, progress))

    def finish(self, result) -> None:
        with self._lock:
            self.times[self.stage] = time.time() - self._stage_t0
            self.progress = 1.0
            self.result = result
            self.status = "done"

    def fail(self, message: str) -> None:
        with self._lock:
            self.error = message
            self.status = "error"

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "jobId": self.id,
                "kind": self.kind,
                "status": self.status,
                "stage": self.stage,
                "stages": self.stages,
                "progress": round(self.progress, 4),
                "times": self.times,
                "result": self.result if self.status == "done" else None,
                "error": self.error,
            }


_jobs: Dict[str, Job] = {}
_pool = ThreadPoolExecutor(max_workers=config.WORKERS, thread_name_prefix="clipforge")
_JOB_TTL = 6 * 3600


def _prune() -> None:
    cutoff = time.time() - _JOB_TTL
    for jid in [j for j, job in _jobs.items() if job.created < cutoff]:
        _jobs.pop(jid, None)


def submit(kind: str, stages: List[str], fn: Callable[[Job], dict]) -> Job:
    _prune()
    job = Job(kind, stages)
    _jobs[job.id] = job

    def runner() -> None:
        job.status = "running"
        try:
            job.finish(fn(job))
        except (ValueError, FileNotFoundError, RuntimeError) as e:
            job.fail(str(e))
        except Exception as e:  # noqa: BLE001 - laporkan apa pun ke klien
            traceback.print_exc()
            job.fail(f"Kesalahan tak terduga: {e.__class__.__name__}: {e}")

    _pool.submit(runner)
    return job


def get(job_id: str) -> Job | None:
    return _jobs.get(job_id)
