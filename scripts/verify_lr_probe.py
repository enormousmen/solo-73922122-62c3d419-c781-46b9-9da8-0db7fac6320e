#!/usr/bin/env python3
"""End-to-end verification of POST /train/lr-probe.

Spins up real uvicorn servers (like scripts/verify_repro.py) and checks:

  P1  validation        -> non-finite / zero / negative / non-numeric / equal
                           learning rates are rejected; no checkpoint yet is 404
  P2  probe correctness -> both candidates start from the same checkpoint,
                           share the same before-loss, and report the
                           checkpoint's run_id / epoch / batch (paused run)
  P3  no side effects   -> probes while paused do not touch loss history;
                           resuming reproduces the uninterrupted run exactly
  P4  live probes       -> probes while training runs succeed, training keeps
                           progressing, finished run is bit-identical
  P5  epoch boundary    -> a checkpoint at the end of an epoch probes the
                           FIRST batch of the next epoch
  P6  no next batch     -> a finished run has no following batch (409)
  P7  force new run     -> an already-obtained probe result still carries the
                           old run_id; after force, a probe before the new run
                           saved a checkpoint is 404 (never the discarded old
                           point); afterwards only the new run_id is reported;
                           status/losses stay responsive throughout
  P8  unverifiable data -> a legacy checkpoint without a dataset fingerprint is
                           refused (409), and so is a checkpoint whose dataset
                           differs from the loaded one
  P9  docs              -> /docs (Swagger UI) and /train/lr-probe in OpenAPI

Exit code 0 iff every scenario passes.  Run from the repo root:

    python3 scripts/verify_lr_probe.py
"""
from __future__ import annotations

import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
sys.path.insert(0, str(ROOT))  # in-process scenario imports the app package

# Small, fast config; checkpoint_every=1 keeps the saved position close.
CFG = {"epochs": 3, "batch_size": 32, "batch_sleep_seconds": 0.02,
       "checkpoint_every": 1, "checkpoint_sleep_seconds": 0.0}
BPE = (600 + 32 - 1) // 32  # 19 batches per epoch on the shipped dataset

RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, note: str = "") -> None:
    RESULTS.append((name, ok, note))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {note}" if note else ""), flush=True)


# --------------------------------------------------------------------- http

def api(port: int, method: str, path: str, body: dict | None = None,
        timeout: float = 120.0) -> tuple[int, dict]:
    url = f"http://127.0.0.1:{port}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return resp.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            return exc.code, json.loads(raw)
        except json.JSONDecodeError:
            return exc.code, {"detail": raw.decode(errors="replace")}


def probe(port: int, a, b) -> tuple[int, dict]:
    return api(port, "POST", "/train/lr-probe", {"lr_a": a, "lr_b": b})


def api_bytes(port: int, path: str) -> bytes:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=60) as resp:
        return resp.read()


# ------------------------------------------------------------------ servers

class Server:
    def __init__(self, runs_dir: Path, log_dir: Path, tag: str,
                 config_path: Path | None = None):
        self.runs_dir = runs_dir
        self.port = _free_port()
        self.tag = tag
        self._log_path = log_dir / f"server-{tag}.log"
        log = open(self._log_path, "wb")
        env = dict(os.environ)
        env["RUNS_DIR"] = str(runs_dir)
        env["CONFIG_PATH"] = str(config_path or (ROOT / "config" / "default.json"))
        env["ENABLE_TEST_CRASH_ENDPOINT"] = "1"
        self.proc = subprocess.Popen(
            [PY, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1",
             "--port", str(self.port)],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        _wait_ready(self.port, self.proc, self._log_path)

    def kill(self) -> None:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{self.port}/test/crash",
                                   data=b"", timeout=5)
        except Exception:
            pass
        self.proc.wait(timeout=10)

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.kill()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_ready(port: int, proc: subprocess.Popen, log_path: Path) -> None:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"server exited early; log:\n"
                + log_path.read_text(errors="replace")[-3000:])
        try:
            code, _ = api(port, "GET", "/train/status", timeout=2)
            if code == 200:
                return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(0.3)
    raise RuntimeError("server did not become ready")


# ------------------------------------------------------------------ helpers

def wait_state(port: int, want: str, timeout: float = 180.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        code, st = api(port, "GET", "/train/status")
        if code == 200 and st["state"] == want:
            return st
        time.sleep(0.05)
    raise RuntimeError(f"state did not become {want!r} within {timeout}s")


def wait_until(port: int, pred, timeout: float = 60.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        code, st = api(port, "GET", "/train/status")
        if code == 200 and pred(st):
            return st
        time.sleep(0.02)
    raise RuntimeError("condition not met within timeout")


def start_run(port: int, cfg: dict, force: bool = True) -> dict:
    code, st = api(port, "POST", "/train/start", {"force": force, "config": cfg})
    assert code == 200, f"start failed: {st}"
    return st


def run_to_finish(port: int, cfg: dict) -> tuple[list, bytes]:
    start_run(port, cfg)
    st = wait_state(port, "finished")
    assert st["completed_batches"] == st["total_batches"], st
    _, losses = api(port, "GET", "/train/losses")
    return losses["losses"], api_bytes(port, "/train/parameters")


def pause_at(port: int, completed: int) -> dict:
    wait_until(port, lambda s: s["completed_batches"] >= completed
               and s["last_checkpoint_at"] is not None)
    code, st = api(port, "POST", "/train/pause")
    assert code == 200 and st["state"] == "paused" and not st["checkpoint_dirty"], st
    return st


def params_equal(a: bytes, b: bytes) -> bool:
    sa = torch.load(io.BytesIO(a), weights_only=True)
    sb = torch.load(io.BytesIO(b), weights_only=True)
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


# ----------------------------------------------------------------- scenarios

def scenario_probe_math() -> None:
    """In-process: a probe candidate IS one real training step; live state intact.

    Reproduces the would-be next training step independently (global torch RNG
    restored from the checkpoint, ordinary ``nn.Dropout``) and compares it to
    the probe's candidate at the checkpoint's own learning rate.  Also checks
    the probe leaves the manager's live model/optimizer/RNG and history alone.
    """
    print("\n== P10: probe math matches a real training step; live state untouched ==")
    import copy

    import torch.nn.functional as F

    from app.checkpoint import load_checkpoint
    from app.config import load_default_config
    from app.model import build_model
    from app.trainer import JobManager, epoch_permutation

    runs = Path(tempfile.mkdtemp(prefix="probe-math-"))
    mgr = JobManager(runs_dir=str(runs), default_config=load_default_config())
    mgr.start({"epochs": 2, "batch_size": 32, "batch_sleep_seconds": 0.0,
               "checkpoint_every": 1})
    while mgr.status()["completed_batches"] < 3:
        time.sleep(0.005)
    mgr.pause()
    assert mgr.status()["state"] == "paused"

    ckpt = load_checkpoint(runs)
    cfg = mgr.cfg
    epoch, nb = int(ckpt["epoch"]), int(ckpt["next_batch"])
    lr = cfg.lr

    live_params = {k: v.clone() for k, v in mgr.model.state_dict().items()}
    live_opt = copy.deepcopy(mgr.optimizer.state_dict())
    live_rng = torch.get_rng_state().clone()
    live_losses = list(mgr.losses)

    code_ok = True
    try:
        resp = mgr.lr_probe(lr, lr * 3)
    except Exception as exc:  # noqa: BLE001
        code_ok = False
        resp = {"error": repr(exc)}
    record("P10 probe runs in process", code_ok, str(resp)[:80] if not code_ok else "")

    rng_unchanged = torch.equal(torch.get_rng_state(), live_rng)
    model_unchanged = all(torch.equal(mgr.model.state_dict()[k], v)
                          for k, v in live_params.items())
    opt_after = mgr.optimizer.state_dict()
    opt_unchanged = (opt_after["param_groups"] == live_opt["param_groups"]
                     and all(torch.equal(opt_after["state"][i][n], live_opt["state"][i][n])
                             for i in opt_after["state"] for n in opt_after["state"][i]))
    record("P10 probe leaves online model parameters untouched", model_unchanged)
    record("P10 probe leaves the online Adam state untouched", opt_unchanged)
    record("P10 probe leaves the global torch RNG stream untouched", rng_unchanged)
    record("P10 probe leaves the loss history untouched", mgr.losses == live_losses)

    if not code_ok:
        return
    # Independent reproduction of the real next training step at lr.
    model = build_model(mgr.input_dim, cfg.hidden_dim, cfg.dropout_p,
                        mgr.num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    opt.load_state_dict(copy.deepcopy(ckpt["optimizer_state_dict"]))
    for group in opt.param_groups:
        group["lr"] = lr
    torch.set_rng_state(ckpt["rng_state"])
    perm = epoch_permutation(cfg.seed, epoch, mgr.num_samples)
    idx = perm[nb * cfg.batch_size:(nb + 1) * cfg.batch_size]
    xb, yb = mgr.X[idx], mgr.y[idx]
    model.train()
    opt.zero_grad(set_to_none=True)
    F.cross_entropy(model(xb), yb).backward()
    opt.step()
    model.eval()
    with torch.no_grad():
        true_after = float(F.cross_entropy(model(xb), yb).item())

    cand = resp["candidates"][0]
    record("P10 candidate at the checkpoint lr equals one real training step",
           cand["lr"] == lr and cand["eval_loss_after"] == true_after,
           f"probe {cand['eval_loss_after']!r} vs true {true_after!r}")
    record("P10 probe targeted the true next batch",
           resp["epoch"] == epoch and resp["batch"] == nb)


def scenario_force_race() -> None:
    """In-process: force=true while a probe is in flight.

    The probe reads the checkpoint file at its start; a force-start that
    removes the old run and writes a new checkpoint mid-probe must not make
    the in-flight probe return the new run's id, and the next probe must use
    only the new run.  (The checkpoint file itself is read atomically.)
    """
    print("\n== P11: force=new run racing an in-flight probe ==")
    from app.checkpoint import load_checkpoint
    from app.config import load_default_config
    from app.trainer import JobManager

    runs = Path(tempfile.mkdtemp(prefix="probe-force-"))
    mgr = JobManager(runs_dir=str(runs), default_config=load_default_config())
    mgr.start({"epochs": 3, "batch_size": 32, "batch_sleep_seconds": 0.0,
               "checkpoint_every": 1})
    while mgr.status()["completed_batches"] < 3:
        time.sleep(0.005)
    mgr.pause()
    old_run = load_checkpoint(runs)["run_id"]

    results: dict = {}
    started = threading.Event()

    def run_probe():
        started.set()
        try:
            results["probe"] = mgr.lr_probe(0.01, 0.1)
        except Exception as exc:  # noqa: BLE001
            results["error"] = repr(exc)

    th = threading.Thread(target=run_probe)
    th.start()
    started.wait()
    # Force a new run while the probe may be reading/saving state.
    mgr.start({"epochs": 3, "batch_size": 32, "batch_sleep_seconds": 0.0,
               "checkpoint_every": 1, "lr": 0.03}, force=True)
    th.join(timeout=60)
    new_run = mgr.status()["run_id"]
    record("P11 force starts a different run", new_run != old_run)
    if "probe" in results:
        record("P11 in-flight probe still reports the old run_id",
               results["probe"]["run_id"] == old_run, results["probe"]["run_id"])
    else:
        # Acceptable: it raced the removal and cleanly refused (404), never
        # silently returning the new run's id.
        record("P11 in-flight probe refused cleanly instead of using the new run",
               "NotFoundError" in results.get("error", ""))
    # Wait for the new run to save a checkpoint; its probes report new_run.
    deadline = time.monotonic() + 30
    resp = None
    while time.monotonic() < deadline:
        try:
            resp = mgr.lr_probe(0.01, 0.1)
        except Exception:  # noqa: BLE001 - no checkpoint yet
            resp = None
        if resp:
            break
        time.sleep(0.01)
    mgr.pause()
    record("P11 post-force probe reports only the new run_id",
           resp is not None and resp["run_id"] == new_run,
           str(resp and resp["run_id"]))
    record("P11 old result object still says old_run_id",
           "probe" not in results or results["probe"]["run_id"] == old_run)


def scenario_validation(port: int) -> None:
    print("\n== P1: validation ==")
    code, body = probe(port, 0.01, 0.2)
    record("P1 probe before any checkpoint -> 404", code == 404,
           str(body.get("detail"))[:70])
    bad = [
        ("string lr", {"lr_a": "0.1", "lr_b": 0.2}),
        ("zero lr", {"lr_a": 0, "lr_b": 0.2}),
        ("negative lr", {"lr_a": -0.1, "lr_b": 0.2}),
        ("infinite lr", {"lr_a": 1e999, "lr_b": 0.2}),
        ("missing lr_b", {"lr_a": 0.1}),
        ("extra key", {"lr_a": 0.1, "lr_b": 0.2, "unexpected": 1}),
        ("bool lr", {"lr_a": True, "lr_b": 0.2}),
    ]
    for label, body in bad:
        code, resp = api(port, "POST", "/train/lr-probe", body)
        record(f"P1 {label} -> 422", code == 422, str(resp.get("detail"))[:70])


def scenario_probe_shape(port: int, ref: tuple[list, bytes]) -> None:
    print("\n== P2: probe correctness (paused run) ==")
    start_run(port, CFG)
    st = pause_at(port, 5)
    ckpt_epoch, ckpt_next = st["epoch"], st["next_batch"]
    code, r = probe(port, 0.01, 0.2)
    record("P2 probe returns 200", code == 200, str(r)[:100] if code != 200 else "")
    if code != 200:
        return
    record("P2 run_id/epoch/batch match the checkpoint",
           r["run_id"] == st["run_id"] and r["epoch"] == ckpt_epoch
           and r["batch"] == ckpt_next
           and r["checkpoint"]["epoch"] == ckpt_epoch
           and r["checkpoint"]["next_batch"] == ckpt_next,
           f"run_id={r['run_id']} epoch={r['epoch']} batch={r['batch']}")
    a, b = r["candidates"]
    record("P2 both candidates share the same starting eval loss",
           a["eval_loss_before"] == b["eval_loss_before"] == r["eval_loss_before"])
    record("P2 candidate lrs echoed, each applied one update",
           a["lr"] == 0.01 and b["lr"] == 0.2
           and a["eval_loss_after"] != a["eval_loss_before"]
           and b["eval_loss_after"] != b["eval_loss_before"])
    record("P2 lr=0.2 moves the batch loss further than lr=0.01",
           abs(b["delta"]) > abs(a["delta"]),
           f"delta {a['delta']:.4f} vs {b['delta']:.4f}")
    code, r2 = probe(port, 0.01, 0.2)
    record("P2 repeated probe is deterministic", code == 200 and r2 == r)
    code, same = probe(port, 0.05, 0.05)
    record("P2 equal lrs -> 422", code == 422, str(same.get("detail"))[:70])
    code, diverge = probe(port, 0.01, 1e20)
    record("P2 diverging lr (non-finite loss) -> 422", code == 422,
           str(diverge.get("detail"))[:70])


def scenario_no_side_effects(port: int, ref: tuple[list, bytes]) -> None:
    print("\n== P3: probes while paused do not perturb training ==")
    _, before = api(port, "GET", "/train/losses")
    for lr in ((0.001, 0.5), (0.02, 0.3), (0.05, 0.1)):
        code, _ = probe(port, *lr)
        assert code == 200
    code, st = api(port, "GET", "/train/status")
    _, after = api(port, "GET", "/train/losses")
    record("P3 status stays paused and probes appended no loss entries",
           st["state"] == "paused" and after["count"] == before["count"])
    record("P3 loss history untouched by probes",
           after["losses"] == before["losses"] and before["count"] < len(ref[0]))
    api(port, "POST", "/train/resume")
    wait_state(port, "finished")
    _, losses = api(port, "GET", "/train/losses")
    params = api_bytes(port, "/train/parameters")
    record("P3 resumed run == uninterrupted (losses)", losses["losses"] == ref[0])
    record("P3 resumed run == uninterrupted (params)", params_equal(params, ref[1]))


def scenario_live_probes(port: int, ref: tuple[list, bytes]) -> None:
    print("\n== P4: probes during a live run ==")
    start_run(port, CFG)
    wait_until(port, lambda s: s["completed_batches"] >= 3)
    hits = 0
    worst = 0.0
    for _ in range(8):
        t0 = time.monotonic()
        code, _ = probe(port, 0.005, 0.05)
        worst = max(worst, time.monotonic() - t0)
        hits += code == 200
        code_s, _ = api(port, "GET", "/train/status")
        if code_s != 200:
            break
        time.sleep(0.01)
    record("P4 probes succeed while training runs", hits >= 1,
           f"{hits}/8 probes ok, worst latency {worst * 1000:.0f}ms")
    st = wait_state(port, "finished")
    _, losses = api(port, "GET", "/train/losses")
    params = api_bytes(port, "/train/parameters")
    record("P4 live-probed run finished", st["state"] == "finished")
    record("P4 live-probed run == uninterrupted (losses)", losses["losses"] == ref[0])
    record("P4 live-probed run == uninterrupted (params)", params_equal(params, ref[1]))


def scenario_epoch_boundary(port: int) -> None:
    print("\n== P5: checkpoint at the end of an epoch probes the next epoch's first batch ==")
    # With batch_size == dataset size there is exactly one batch per epoch.
    # After that batch the trainer persists (epoch=0, next_batch=1) and only
    # then advances in memory to (1, 0) and starts the next epoch's batch.
    # While that next batch is sleeping, the disk checkpoint is exactly the
    # epoch-boundary one.
    n = 600
    cfg = {"epochs": 3, "batch_size": n, "batch_sleep_seconds": 0.6,
           "checkpoint_every": 1}
    ok = False
    hit = None
    for _ in range(3):
        start_run(port, cfg)
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            _, st = api(port, "GET", "/train/status")
            if st["state"] == "running" and st["epoch"] == 1 and st["next_batch"] == 0:
                code, r = probe(port, 0.01, 0.1)
                hit = (code, r)
                break
            time.sleep(0.01)
        api(port, "POST", "/train/pause")
        _, st = api(port, "GET", "/train/status")
        if hit and hit[0] == 200 and hit[1]["checkpoint"]["epoch"] == 0 \
                and hit[1]["checkpoint"]["next_batch"] == 1:
            ok = True
            break
        # Window missed (or a later checkpoint landed first): retry.
        api(port, "POST", "/train/resume")
        wait_state(port, "finished")
        hit = None
    record("P5 probe at an epoch-boundary checkpoint targets the next epoch's first batch",
           ok and hit[1]["epoch"] == 1 and hit[1]["batch"] == 0,
           f"checkpoint {hit[1].get('checkpoint')} -> batch "
           f"{hit[1].get('epoch')}/{hit[1].get('batch')}" if hit and hit[0] == 200
           else f"probe failed: {hit}")
    # Leave the service with the run finished for the next scenario.
    code, st = api(port, "GET", "/train/status")
    if st["state"] == "paused":
        api(port, "POST", "/train/resume")
    wait_state(port, "finished")


def scenario_no_next_batch(port: int) -> None:
    print("\n== P6: no following batch after the run finished ==")
    wait_state(port, "finished")
    code, r = probe(port, 0.01, 0.2)
    record("P6 probe on a finished run -> 409", code == 409,
           str(r.get("detail"))[:80])


def scenario_force_run_id(port: int, runs_dir: Path) -> None:
    print("\n== P7: force new run: old result kept, new probes use the new run ==")
    start_run(port, CFG)
    st = pause_at(port, 4)
    old_run = st["run_id"]
    code, probe_old = probe(port, 0.01, 0.2)
    record("P7 probe reports the old run_id", code == 200 and probe_old["run_id"] == old_run)
    record("P7 old checkpoint on disk before force", (runs_dir / "checkpoint.pt").exists())

    start_run(port, {**CFG, "lr": 0.02})  # force=true discards the old run
    _, st2 = api(port, "GET", "/train/status")
    new_run = st2["run_id"]
    record("P7 force started a new run", new_run != old_run)

    # Immediately after start, before the new run writes its first checkpoint,
    # an old point must NOT be used even though the request could have the old
    # one in hand: the discarded checkpoint is gone from disk.
    record("P7 old checkpoint removed on force", not (runs_dir / "checkpoint.pt").exists())
    code, resp = probe(port, 0.01, 0.2)
    record("P7 probe before the new run saved a point -> 404 (no fallback to old)",
           code == 404, f"got {code}: {str(resp.get('detail'))[:60]}")

    # An already-obtained response keeps its old run_id (immutable snapshot).
    record("P7 previously obtained result still carries the old run_id",
           probe_old["run_id"] == old_run)

    st = pause_at(port, 6)
    code, resp = probe(port, 0.01, 0.2)
    record("P7 probe after the new run has a checkpoint reports only the new run_id",
           code == 200 and resp["run_id"] == new_run, f"got {resp.get('run_id')}")

    # Queries stay responsive around probes and training still finishes.
    _, status_mid = api(port, "GET", "/train/status")
    record("P7 status stays responsive while the probes ran",
           status_mid["state"] == "paused")
    api(port, "POST", "/train/resume")
    st = wait_state(port, "finished")
    _, losses = api(port, "GET", "/train/losses")
    record("P7 new run finished with its own full history",
           st["state"] == "finished" and losses["count"] == 3 * BPE)


def scenario_unverifiable_dataset(port: int, runs_dir: Path, log_dir: Path) -> None:
    print("\n== P8: checkpoint without a verifiable dataset is refused ==")
    start_run(port, CFG)
    pause_at(port, 2)
    ckpt_file = runs_dir / "checkpoint.pt"
    payload = torch.load(ckpt_file, map_location="cpu", weights_only=True)
    record("P8 checkpoint stores a dataset fingerprint",
           bool((payload.get("dataset") or {}).get("fingerprint")))

    legacy = dict(payload)
    legacy.pop("dataset", None)
    torch.save(legacy, ckpt_file)
    try:
        code, r = probe(port, 0.01, 0.2)
        record("P8 legacy checkpoint without fingerprint -> 409", code == 409,
               str(r.get("detail"))[:80])
    finally:
        torch.save(payload, ckpt_file)

    # Different dataset on disk for a restarted service with the same runs dir.
    other = log_dir / "other.csv"
    _write_other_dataset(other)
    other_runs = log_dir / "other-runs"
    shutil.copytree(runs_dir, other_runs)
    base = json.loads((ROOT / "config" / "default.json").read_text())
    base["dataset_path"] = str(other)
    cfg_path = log_dir / "other-config.json"
    cfg_path.write_text(json.dumps(base))
    server = Server(other_runs, log_dir, "P8-other-dataset", config_path=cfg_path)
    try:
        code, r = probe(server.port, 0.01, 0.2)
        record("P8 checkpoint from a different dataset -> 409", code == 409,
               str(r.get("detail"))[:80])
    finally:
        server.stop()


def _write_other_dataset(path: Path) -> None:
    lines = ["f1,f2,label"]
    for i in range(40):
        lines.append(f"{i * 0.1:.2f},{(i % 7) * 0.3:.2f},{i % 3}")
    path.write_text("\n".join(lines) + "\n")


def scenario_docs(port: int) -> None:
    print("\n== P9: docs ==")
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/docs", timeout=10) as resp:
        html = resp.read().lower()
        status = resp.status
    record("P9 /docs serves Swagger UI", status == 200 and b"swagger" in html)
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/openapi.json", timeout=10) as resp:
        spec = json.loads(resp.read())
    record("P9 /train/lr-probe documented in OpenAPI",
           "/train/lr-probe" in spec.get("paths", {})
           and "post" in spec["paths"]["/train/lr-probe"])


# ---------------------------------------------------------------------- main

def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="verify-lr-probe-"))
    runs_dir = work / "runs"
    log_dir = work / "logs"
    log_dir.mkdir()
    print(f"workdir: {work}")
    server = Server(runs_dir, log_dir, "main")
    try:
        scenario_validation(server.port)
        print("\n== reference: uninterrupted run ==")
        ref = run_to_finish(server.port, CFG)
        record("reference run finished", len(ref[0]) == 3 * BPE,
               f"{len(ref[0])} batches")
        scenario_probe_shape(server.port, ref)
        scenario_no_side_effects(server.port, ref)
        scenario_live_probes(server.port, ref)
        scenario_epoch_boundary(server.port)
        scenario_no_next_batch(server.port)
        scenario_force_run_id(server.port, runs_dir)
        scenario_unverifiable_dataset(server.port, runs_dir, log_dir)
        scenario_docs(server.port)
        scenario_probe_math()
        scenario_force_race()
    finally:
        server.stop()

    print("\n================ SUMMARY ================")
    failed = 0
    for name, ok, note in RESULTS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        failed += 0 if ok else 1
    print(f"  {len(RESULTS) - failed}/{len(RESULTS)} checks passed")
    if failed:
        print(f"  server logs under {log_dir}")
    else:
        shutil.rmtree(work, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
