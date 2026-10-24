"""A cross-process lease on the GPU.

There is one 6GB card and two pipelines that each want ~2GB of model plus
working memory: the batch pass (large-v3 + pyannote) and the realtime server
(turbo). Loading the second while the first holds the card does not fail
cleanly -- it hangs. Until this module existed the rule "don't run both" lived
only in the README.

The lease is an flock, so the kernel holds it and it is released automatically
if a process dies. Holder details are written into the file purely so the loser
can say something useful about who is winning.

With no NVIDIA GPU (a MacBook, a CPU-only box) there is no card to fight over:
acquire() then returns a lease that holds nothing, instantly, and never touches
the lockfile.

Two shapes, because the two callers differ:
    with lease("batch"): ...        # scoped, for a run that ends
    handle = acquire("realtime")    # open-ended, for a server that holds a
    handle.release()                # model loaded across many requests
"""
import fcntl
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

from core import device

LOCK_PATH = Path(os.environ.get("SUPERMEET_GPU_LOCK", "/tmp/supermeet-gpu.lock"))


class GPUBusy(RuntimeError):
    """The lease is held elsewhere and we chose not to wait."""

    def __init__(self, holder: dict | None):
        self.holder = holder or {}
        who = self.holder.get("name", "another process")
        pid = self.holder.get("pid", "?")
        since = self.holder.get("since")
        held = f" for {time.time() - since:.0f}s" if since else ""
        super().__init__(f"GPU is leased by {who} (pid {pid}){held}")


def describe(choice: device.Choice) -> str:
    """One line for the log: which model runs on what, and why not the GPU."""
    note = choice.compute_type
    if not choice.is_cuda:
        forced = os.environ.get("SUPERMEET_DEVICE", "").strip().lower() == "cpu"
        note = "dipaksa SUPERMEET_DEVICE=cpu" if forced else "tidak ada GPU NVIDIA"
    return f"{choice.model} on {choice.device} ({note})"


def holder() -> dict | None:
    """Who holds the lease right now, or None if nobody does / unreadable."""
    try:
        data = json.loads(LOCK_PATH.read_text() or "{}")
    except (OSError, ValueError):
        return None
    return data or None


class Lease:
    """An acquired GPU lease. Release it, or let the process exit."""

    def __init__(self, name: str, fh):
        self.name = name
        self._fh = fh

    def release(self):
        if self._fh is None:
            return
        try:
            self._fh.seek(0)
            self._fh.truncate()
            self._fh.flush()
        except OSError:
            pass
        self._fh.close()      # releases the flock
        self._fh = None

    @property
    def held(self) -> bool:
        return self._fh is not None


def acquire(name: str, wait: bool = True, timeout: float | None = None,
            on_wait=None) -> Lease:
    """Take the lease.

    wait=False raises GPUBusy immediately if someone else holds it -- what the
    watcher wants, so the file simply stays in the inbox until the next poll.
    wait=True blocks, optionally up to `timeout` seconds, then raises GPUBusy --
    what a person starting a meeting wants. on_wait(holder) fires once if we
    actually have to wait, so the caller can explain the pause.
    """
    if not device.cuda_available():
        return Lease(name, None)    # nothing to lease; held is False
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    fh = open(LOCK_PATH, "a+")
    try:
        _flock(fh, wait=wait, timeout=timeout, on_wait=on_wait)
    except BaseException:
        fh.close()
        raise

    fh.seek(0)
    fh.truncate()
    fh.write(json.dumps({"name": name, "pid": os.getpid(), "since": time.time()}))
    fh.flush()
    return Lease(name, fh)


def _flock(fh, wait: bool, timeout: float | None, on_wait):
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return
    except BlockingIOError:
        pass

    current = holder()
    if not wait:
        raise GPUBusy(current) from None
    if on_wait:
        on_wait(current)

    deadline = time.monotonic() + timeout if timeout else None
    while True:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            if deadline and time.monotonic() > deadline:
                raise GPUBusy(holder()) from None
            time.sleep(1.0)


@contextmanager
def lease(name: str, wait: bool = True, timeout: float | None = None,
          on_wait=None):
    """Scoped version of acquire(), for a run with a definite end."""
    handle = acquire(name, wait=wait, timeout=timeout, on_wait=on_wait)
    try:
        yield handle
    finally:
        handle.release()
