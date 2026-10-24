#!/usr/bin/env python3
"""Windows-side audio capture for supermeet.

Runs on the WINDOWS python (not the WSL one) because only Windows exposes WASAPI
loopback — the speaker output, i.e. everyone else on the call. Mixes that with
the microphone (you) and streams 16kHz mono PCM to the server over a websocket.

    py -m pip install soundcard numpy websockets
    py win_capture.py --server ws://localhost:8848/ws/ingest

Sources:
    --source both      speakers + mic (default; what you want on a video call)
    --source loopback  speakers only
    --source mic       mic only (use this for an in-person meeting)
"""
import argparse
import queue
import sys
import threading
import time

import numpy as np

SAMPLE_RATE = 16000
BLOCK = 1600  # 100ms


def _record(device, out_q: queue.Queue, stop: threading.Event, gain: float, label: str):
    import soundcard as sc
    try:
        with device.recorder(samplerate=SAMPLE_RATE, channels=1, blocksize=BLOCK) as rec:
            print(f"  [{label}] recording: {device.name}", flush=True)
            while not stop.is_set():
                data = rec.record(numframes=BLOCK)
                mono = data[:, 0] if data.ndim > 1 else data
                try:
                    out_q.put_nowait(mono.astype(np.float32) * gain)
                except queue.Full:
                    pass  # server or network is behind; dropping is better than lagging
    except Exception as e:
        print(f"  [{label}] stopped: {e}", file=sys.stderr, flush=True)
        stop.set()


def _drain(q: queue.Queue) -> np.ndarray | None:
    try:
        return q.get(timeout=0.5)
    except queue.Empty:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="ws://localhost:8848/ws/ingest")
    ap.add_argument("--source", choices=["both", "loopback", "mic"], default="both")
    ap.add_argument("--mic-gain", type=float, default=1.0)
    ap.add_argument("--speaker-gain", type=float, default=1.0)
    ap.add_argument("--list", action="store_true", help="list audio devices and exit")
    args = ap.parse_args()

    import soundcard as sc
    from websockets.sync.client import connect

    if args.list:
        print("Speakers:")
        for s in sc.all_speakers():
            print(f"  {s.name}")
        print("Microphones:")
        for m in sc.all_microphones():
            print(f"  {m.name}")
        return

    stop = threading.Event()
    queues, threads = [], []

    if args.source in ("both", "loopback"):
        speaker = sc.default_speaker()
        loop_dev = sc.get_microphone(speaker.name, include_loopback=True)
        q = queue.Queue(maxsize=50)
        queues.append(q)
        threads.append(threading.Thread(
            target=_record, args=(loop_dev, q, stop, args.speaker_gain, "speakers"), daemon=True))

    if args.source in ("both", "mic"):
        q = queue.Queue(maxsize=50)
        queues.append(q)
        threads.append(threading.Thread(
            target=_record, args=(sc.default_microphone(), q, stop, args.mic_gain, "mic"),
            daemon=True))

    for t in threads:
        t.start()

    print(f"connecting to {args.server} ...", flush=True)
    sent = 0
    try:
        with connect(args.server, max_queue=None) as ws:
            print("connected. Ctrl+C to stop.\n", flush=True)
            while not stop.is_set():
                blocks = [b for b in (_drain(q) for q in queues) if b is not None]
                if not blocks:
                    continue
                n = min(len(b) for b in blocks)
                mixed = np.sum([b[:n] for b in blocks], axis=0)
                # Summing two sources can clip; scale back only when it would.
                peak = np.abs(mixed).max()
                if peak > 1.0:
                    mixed = mixed / peak
                ws.send((mixed * 32767).astype(np.int16).tobytes())
                sent += n
                if sent % (SAMPLE_RATE * 10) < BLOCK:
                    print(f"  streamed {sent / SAMPLE_RATE:.0f}s", flush=True)
    except KeyboardInterrupt:
        print("\nstopping...")
    finally:
        stop.set()
        time.sleep(0.3)


if __name__ == "__main__":
    main()
