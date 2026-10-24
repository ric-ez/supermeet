"""FastAPI server: audio in over one websocket, live meeting state out over another."""
import asyncio
import contextlib
import json
import logging

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from . import config, kuliah_web, stt
from .brain import Brain
from .kuliah_web import router as kuliah_router
from .session import Session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-12s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rtsm.app")


class Hub:
    """Holds the one active meeting and fans state out to every viewer."""

    def __init__(self):
        self.session: Session | None = None
        self.stt: stt.StreamingSTT | None = None
        self.brain: Brain | None = None
        self.clients: set[WebSocket] = set()
        self.ingest_count = 0
        self._handoff_tasks: set[asyncio.Task] = set()
        self._commands: set[asyncio.Task] = set()
        # Set whenever no archival handoff is occupying the GPU.
        self._handoff_idle = asyncio.Event()
        self._handoff_idle.set()
        self._lock = asyncio.Lock()

    def spawn(self, coro, name: str) -> asyncio.Task:
        """Run a command detached from the websocket that asked for it."""
        task = asyncio.create_task(coro, name=name)
        task.add_done_callback(_log_task_result)
        self._commands.add(task)
        task.add_done_callback(self._commands.discard)
        return task

    # --- fanout ----------------------------------------------------------
    async def publish(self, event: dict):
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_json(event)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)

    def status(self) -> dict:
        s = self.session
        return {
            "type": "status",
            "recording": s is not None,
            "session_id": s.id if s else None,
            "title": s.title if s else None,
            "ai": s.ai if s else None,
            "elapsed": s.elapsed if s else 0,
            "utterances": len(s.utterances) if s else 0,
            "ingest_connected": self.ingest_count,
            "model": config.WHISPER_MODEL,
            "claude_model": config.CLAUDE_MODEL,
            "cost": round(self.brain.cost, 4) if self.brain else 0.0,
        }

    # --- lifecycle -------------------------------------------------------
    async def _on_partial(self, text: str):
        await self.publish({"type": "partial", "text": text})

    async def _on_final(self, u: stt.Utterance):
        session = self.session
        if session is None:
            return  # stopped while the last utterance was still transcribing
        session.add_utterance(u.start, u.end, u.text)
        await self.publish({"type": "final", "utterance": u.as_dict()})

    async def _on_brain(self, event: dict):
        session = self.session
        if session is not None:
            if event.get("type") == "summary":
                session.summary = event["data"]
            elif event.get("type") == "questions":
                session.questions = event["data"].get("questions", [])
        await self.publish(event)

    async def _await_handoff(self):
        """Block until the archival pass has released the GPU.

        The batch pass holds large-v3 plus pyannote; loading turbo on top of
        that on a 6GB card never returns. Waiting is the only safe order.

        This is the in-process half of the interlock — core/gpu.py handles the
        cross-process case, for a watcher run we did not start ourselves.
        """
        if self._handoff_idle.is_set():
            return
        await self.publish({"type": "waiting", "reason": "handoff",
                            "detail": "nunggu pipeline arsip ngelepas GPU"})
        try:
            await asyncio.wait_for(self._handoff_idle.wait(),
                                   timeout=config.HANDOFF_WAIT_TIMEOUT)
        except asyncio.TimeoutError:
            log.warning("handoff still running after %.0fs — starting anyway",
                        config.HANDOFF_WAIT_TIMEOUT)

    async def start(self, title: str | None, ai: bool = True):
        if self.session:
            return self.session
        # Outside the lock on purpose: this can wait minutes for the batch pass,
        # and holding the lock through it would make stop() unresponsive too.
        await self._await_handoff()
        if kuliah_web.running():
            await self.publish({"type": "waiting", "reason": "kuliah",
                                "detail": "ngehentiin transkrip kuliah dulu biar GPU-nya bebas"})
            await kuliah_web.stop_for_live()
        async with self._lock:
            if self.session:
                return self.session

            # Get the engine running BEFORE creating the session, so a failed
            # start leaves no orphan session directory behind.
            engine = stt.StreamingSTT(self._on_partial, self._on_final)
            try:
                async with asyncio.timeout(config.MODEL_LOAD_TIMEOUT):
                    await engine.start()
            except (asyncio.TimeoutError, TimeoutError):
                log.error("model load exceeded %.0fs", config.MODEL_LOAD_TIMEOUT)
                await self.publish({"type": "start_failed",
                                    "detail": "model nggak selesai load — cek VRAM"})
                return None
            except Exception as e:
                log.exception("engine start failed")
                await self.publish({"type": "start_failed", "detail": str(e)[:200]})
                return None

            session = Session(title, ai=ai)
            brain = None
            if ai:
                brain = Brain(session.transcript_text, self._on_brain)
                await brain.start()

            self.session, self.stt, self.brain = session, engine, brain
            await self.publish(self.status())
            return session

    async def stop(self, run_handoff: bool):
        async with self._lock:
            session, engine, brain = self.session, self.stt, self.brain
            if not session:
                return None
            self.session = self.stt = self.brain = None

        await engine.stop()
        if brain:
            await brain.stop()
        paths = session.close()
        await self.publish({"type": "stopped", "session_id": session.id, "paths": paths})

        if run_handoff:
            self._handoff_idle.clear()
            task = asyncio.create_task(self._handoff(session), name=f"handoff-{session.id}")
            task.add_done_callback(_log_task_result)
            self._handoff_tasks.add(task)
            task.add_done_callback(self._handoff_tasks.discard)
        return {"session_id": session.id, "paths": paths,
                "cost": round(brain.cost, 4) if brain else 0.0}

    async def _handoff(self, session: Session):
        # The batch pass loads large-v3 + pyannote; turbo has to be out of the
        # way first or the 6GB card OOMs. Releasing the model also drops the GPU
        # lease, which is how the batch subprocess gets in. start() waits on
        # _handoff_idle for the same reason, from the other direction.
        try:
            await stt.release_model()
            await self.publish({"type": "handoff", "state": "running",
                                "session_id": session.id})
            result = await session.handoff()
            await self.publish({"type": "handoff", "state": result,
                                "session_id": session.id})
        finally:
            self._handoff_idle.set()


hub = Hub()
kuliah_web.live_active = lambda: hub.session is not None


def _log_task_result(task: asyncio.Task):
    if not task.cancelled() and task.exception():
        log.error("background task %s failed", task.get_name(), exc_info=task.exception())


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    config.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    # Warm the model now so hitting Record doesn't cost 20 seconds. The
    # reference matters: asyncio only holds a weak one, so a bare create_task
    # can be garbage-collected mid-load — silently, with the exception eaten.
    warm = asyncio.create_task(stt.warm_up(), name="warm-up")
    warm.add_done_callback(_log_task_result)
    app.state.warm_up_task = warm
    yield
    if hub.session:
        await hub.stop(run_handoff=False)


app = FastAPI(title="supermeet", lifespan=lifespan)
app.include_router(kuliah_router)


@app.get("/")
async def index():
    return FileResponse(config.WEB_DIR / "index.html")


@app.get("/ca.crt")
async def ca_cert():
    """The local CA from make-cert.sh, for installing on an iPad (see README).
    Safari only offers to install it as a profile when served with this type."""
    ca = config.PROJECT_DIR / "certs" / "ca.pem"
    if not ca.is_file():
        raise HTTPException(404, "belum ada CA — jalanin ./supermeet.py serve --https")
    return FileResponse(ca, media_type="application/x-x509-ca-cert")


@app.get("/api/status")
async def api_status():
    return JSONResponse(hub.status())


@app.websocket("/ws/ingest")
async def ws_ingest(ws: WebSocket):
    """Raw 16kHz mono int16 PCM frames. Any capture client can speak this."""
    await ws.accept()
    hub.ingest_count += 1
    await hub.publish(hub.status())
    log.info("ingest connected (%d total)", hub.ingest_count)
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            pcm = msg.get("bytes")
            if not pcm:
                continue
            session, engine = hub.session, hub.stt
            if session is None or engine is None:
                continue  # not recording — drop the audio, keep the socket
            session.write_audio(pcm)
            await engine.feed(pcm)
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("ingest socket error")
    finally:
        hub.ingest_count = max(0, hub.ingest_count - 1)
        log.info("ingest disconnected (%d left)", hub.ingest_count)
        await hub.publish(hub.status())


@app.websocket("/ws/client")
async def ws_client(ws: WebSocket):
    await ws.accept()
    hub.clients.add(ws)
    await ws.send_json(hub.status())
    if hub.session:
        await ws.send_json({"type": "restore", "state": hub.session.state()})
    try:
        while True:
            cmd = json.loads(await ws.receive_text())
            action = cmd.get("cmd")
            # start/stop run detached. Loading the model starves the event loop
            # for ~15s, which can trip this socket's keepalive; if the command
            # were awaited inline, that disconnect would cancel the session
            # start or stop half-way through.
            if action == "start":
                hub.spawn(hub.start(cmd.get("title"), bool(cmd.get("ai", True))), "start")
            elif action == "stop":
                hub.spawn(hub.stop(run_handoff=bool(cmd.get("handoff", True))), "stop")
            elif action == "refresh" and hub.brain:
                await hub.brain.request_now(cmd.get("kind", "questions"))
            elif action == "ping":
                await ws.send_json(hub.status())
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("client socket error")
    finally:
        hub.clients.discard(ws)


def main():
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="warning")


if __name__ == "__main__":
    main()
