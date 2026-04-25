"""
app.py
FastAPI application for NWP v11 -- hybrid LSTM + n-gram federated next-word
prediction server.

Responsibilities
----------------
- Serve the Frontend static files and WebSocket push channel.
- Expose REST endpoints for prediction, session management, model registry,
  personalisation, federated sync, peer discovery, and LSTM tuning.
- Run three background asyncio tasks:
    _auto_discovery_loop : Scan for peers every 60 s.
    _gossip_loop         : Run gossip FL rounds when peers are available.
    _flush_loop          : Drain inbound peer delta queue every 30 s.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from typing import Any, Literal

import httpx
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, WebSocket
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from nwp_v12.FL.gossip import GossipEngine, PeerReputation
from nwp_v12.FL.federated_sync import FederatedSyncAgent
from nwp_v12.FL.training_agent import TrainingAgent
from .db import SqliteDB, init_db, json_dumps, json_loads, now_ts
from .dl_module.hashing import sha256_hex
from .dl_module.redact import redact_text
from .model_registry import ModelRegistry
from .peer_discovery import DiscoveredPeer, PeerDiscovery
from .settings import TrainerSettings
from .ws import WebSocketHub

# Suppress Pydantic warning about 'model_' prefixed field names.
_NS = {"protected_namespaces": ()}


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class StatusResponse(BaseModel):
    model_config = _NS
    ok: bool = True
    device_id: str
    discovered_peers: int = 0


class PredictRequest(BaseModel):
    model_config = _NS
    model_id: str = Field(min_length=1)
    context_text: str = ""
    cursor_pos: int = 0
    k: int = Field(default=5, ge=1, le=5)
    max_chars: int = Field(default=24, ge=1, le=64)


class SessionStartRequest(BaseModel):
    model_config = _NS
    model_id: str = Field(min_length=1)


class SessionEventRequest(BaseModel):
    model_config = _NS
    session_id: str = Field(min_length=8)
    ts: int = Field(default_factory=now_ts)
    type: str = Field(min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)


class SessionEndRequest(BaseModel):
    model_config = _NS
    session_id: str = Field(min_length=8)
    final_text: str = ""


class TrainRequest(BaseModel):
    model_config = _NS
    model_id: str = Field(min_length=1)
    reason: Literal["manual", "session_end", "scheduled"] = "manual"


class ShareRequest(BaseModel):
    model_config = _NS
    model_id: str = Field(min_length=1)
    peers: list[str] | None = None
    mode: Literal["manual", "scheduled"] = "manual"


class PullGlobalRequest(BaseModel):
    model_config = _NS
    model_id: str = Field(min_length=1)


class GossipRoundRequest(BaseModel):
    model_config = _NS
    model_id: str = Field(min_length=1)
    peers: list[str] | None = None


class SettingsResponse(BaseModel):
    model_config = _NS
    ok: bool = True
    auto_train: bool
    auto_share: bool
    max_concurrent_peer_sync: int
    discovery_enable_lan: bool
    gossip_enabled: bool
    gossip_interval_s: int
    lstm_conf_threshold: float
    lstm_weight: float
    local_ngram_weight: float
    global_ngram_weight: float
    local_ngram_fallback_weight: float
    global_ngram_fallback_weight: float
    lstm_train_steps: int


class SettingsUpdateRequest(BaseModel):
    model_config = _NS
    auto_train: bool | None = None
    auto_share: bool | None = None
    max_concurrent_peer_sync: int | None = None
    discovery_enable_lan: bool | None = None
    gossip_enabled: bool | None = None
    gossip_interval_s: int | None = None
    lstm_conf_threshold: float | None = None
    lstm_weight: float | None = None
    local_ngram_weight: float | None = None
    global_ngram_weight: float | None = None
    local_ngram_fallback_weight: float | None = None
    global_ngram_fallback_weight: float | None = None
    lstm_train_steps: int | None = None


class ObserveTextRequest(BaseModel):
    model_config = _NS
    model_id: str = Field(min_length=1)
    text: str = ""


class UserWordRequest(BaseModel):
    model_config = _NS
    model_id: str = Field(min_length=1)
    word: str = Field(min_length=1)
    weight: float = 1.0
    category: str = "custom"


class CreateModelRequest(BaseModel):
    model_config = _NS
    name: str = Field(min_length=1, max_length=80)
    creation_type: Literal["default", "blank", "clone"] = "blank"
    source_model_id: str | None = None


class UpdateModelRequest(BaseModel):
    model_config = _NS
    model_id: str = Field(min_length=1)
    name: str | None = None
    enabled: bool | None = None


class DeleteModelRequest(BaseModel):
    model_config = _NS
    model_id: str = Field(min_length=1)


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app(*, settings: TrainerSettings, static_dir: str) -> FastAPI:
    """
    Construct and return the FastAPI application.

    Parameters
    ----------
    settings   : Immutable runtime configuration (port, data dir, device ID).
    static_dir : Path to the Frontend/ directory containing index.html etc.
    """
    app = FastAPI(title="NWP P2P Federated - v11 Hybrid LSTM", version="11.0")

    db = SqliteDB(settings.db_path)
    init_db(db)
    hub = WebSocketHub()
    registry = ModelRegistry(settings.data_dir)
    reputation = PeerReputation()
    gossip_engine = GossipEngine(own_url=settings.own_url, reputation=reputation)
    disc = PeerDiscovery(own_url=settings.own_url)
    federated_sync = FederatedSyncAgent(device_id=settings.device_id)

    def get_db() -> SqliteDB:
        return db

    # -- Config helpers -------------------------------------------------------

    def _cfg_get(key: str, default: Any = None) -> Any:
        row = db.query_one("SELECT value_json FROM config WHERE key=?;", (key,))
        if not row:
            return default
        try:
            return json_loads(row["value_json"])
        except Exception:
            return default

    def _cfg_set(key: str, value: Any) -> None:
        db.execute(
            "INSERT INTO config(key, value_json) VALUES(?, ?)"
            " ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json;",
            (key, json_dumps(value)),
        )

    # -- Startup helpers ------------------------------------------------------

    def _ensure_defaults() -> None:
        """Create the 'default' model row if it does not exist."""
        row = db.query_one("SELECT model_id FROM models_registry WHERE model_id='default';")
        if not row:
            db.execute(
                "INSERT OR IGNORE INTO models_registry"
                "(model_id, name, enabled, created_at, creation_type)"
                " VALUES('default', 'Default', 1, ?, 'default');",
                (now_ts(),),
            )
        registry.get("default")

    def _load_active_models() -> None:
        """Pre-load all enabled model slots so they are warm on first request."""
        rows = db.query_all("SELECT model_id FROM models_registry WHERE enabled=1;")
        for r in rows:
            registry.get(r["model_id"])

    # -- Background tasks -----------------------------------------------------

    async def _auto_discovery_loop() -> None:
        """Scan for peers every 60 s and broadcast results over WebSocket."""
        await asyncio.sleep(5)
        while True:
            try:
                enable_lan = bool(_cfg_get("discovery_enable_lan", False))
                found = await disc.scan(enable_lan=enable_lan)
                for p in found:
                    reputation.upsert(p.url, peer_id=p.peer_id)
                if found:
                    await hub.broadcast_json({
                        "type": "peers_discovered",
                        "count": len(found),
                        "peers": [p.to_dict() for p in found],
                    })
            except asyncio.CancelledError:
                return
            except Exception:
                pass
            await asyncio.sleep(60)

    async def _gossip_loop() -> None:
        """Run gossip FL rounds at the configured interval when peers exist."""
        await asyncio.sleep(15)
        interval = 300
        while True:
            try:
                if bool(_cfg_get("gossip_enabled", True)):
                    interval = int(_cfg_get("gossip_interval_s", 300))
                    peer_urls = disc.get_active_urls()
                    if peer_urls:
                        rows = db.query_all(
                            "SELECT model_id FROM models_registry WHERE enabled=1;"
                        )
                        for row in rows:
                            mid = row["model_id"]
                            result = await gossip_engine.run_round(
                                model_id=mid,
                                peer_urls=peer_urls,
                                get_slot_fn=registry.get,
                            )
                            if result.get("ok"):
                                await hub.broadcast_json({
                                    "type": "gossip_round_complete",
                                    "model_id": mid,
                                    **result,
                                })
            except asyncio.CancelledError:
                return
            except Exception:
                pass
            await asyncio.sleep(interval)

    async def _flush_loop() -> None:
        """Drain inbound peer delta queues into global models every 30 s."""
        while True:
            try:
                rows = db.query_all(
                    "SELECT model_id FROM models_registry WHERE enabled=1;"
                )
                for row in rows:
                    slot = registry.get(row["model_id"])
                    merged = federated_sync.flush_into_global(slot.model_agent)
                    if merged > 0:
                        await hub.broadcast_json({
                            "type": "global_model_updated",
                            "model_id": row["model_id"],
                            "deltas_merged": merged,
                        })
            except asyncio.CancelledError:
                return
            except Exception:
                pass
            await asyncio.sleep(30)

    @app.on_event("startup")
    async def _startup() -> None:
        _ensure_defaults()
        os.makedirs(settings.data_dir, exist_ok=True)
        _load_active_models()
        max_sync = int(_cfg_get("max_concurrent_peer_sync", 10))
        app.state.share_semaphore = asyncio.Semaphore(max_sync)
        asyncio.create_task(_auto_discovery_loop())
        asyncio.create_task(_gossip_loop())
        asyncio.create_task(_flush_loop())

    # -- Static / UI ----------------------------------------------------------

    @app.get("/")
    async def index() -> FileResponse:
        path = os.path.join(static_dir, "index.html")
        if not os.path.exists(path):
            raise HTTPException(status_code=500, detail="UI not built")
        return FileResponse(path)

    @app.get("/static/{path:path}")
    async def static_files(path: str) -> FileResponse:
        full = os.path.abspath(os.path.join(static_dir, path))
        if not full.startswith(os.path.abspath(static_dir)):
            raise HTTPException(status_code=400, detail="bad path")
        if not os.path.exists(full):
            raise HTTPException(status_code=404, detail="not found")
        return FileResponse(full)

    @app.websocket("/ws/local")
    async def ws_local(ws: WebSocket) -> None:
        await hub.connect(ws)
        try:
            while True:
                await ws.receive_text()
        except Exception:
            pass
        finally:
            await hub.disconnect(ws)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "ok": True,
            "peer_id": settings.device_id,
            "version": "11.0",
            "engine": "hybrid-lstm-v11",
        }

    # -- Status / Settings ----------------------------------------------------

    @app.get("/local/status")
    async def status() -> StatusResponse:
        return StatusResponse(
            device_id=settings.device_id,
            discovered_peers=len(disc.peers),
        )

    @app.get("/local/settings")
    async def get_settings() -> SettingsResponse:
        from .dl_module.hybrid_model import LstmWordModel, HybridModelAgent
        return SettingsResponse(
            auto_train=bool(_cfg_get("auto_train", True)),
            auto_share=bool(_cfg_get("auto_share", False)),
            max_concurrent_peer_sync=int(_cfg_get("max_concurrent_peer_sync", 10)),
            discovery_enable_lan=bool(_cfg_get("discovery_enable_lan", False)),
            gossip_enabled=bool(_cfg_get("gossip_enabled", True)),
            gossip_interval_s=int(_cfg_get("gossip_interval_s", 300)),
            lstm_conf_threshold=float(
                _cfg_get("lstm_conf_threshold", LstmWordModel.CONF_THRESHOLD)
            ),
            lstm_weight=float(_cfg_get("lstm_weight", HybridModelAgent.LSTM_WEIGHT)),
            local_ngram_weight=float(
                _cfg_get("local_ngram_weight", HybridModelAgent.LOCAL_NGRAM_WEIGHT)
            ),
            global_ngram_weight=float(
                _cfg_get("global_ngram_weight", HybridModelAgent.GLOBAL_NGRAM_WEIGHT)
            ),
            local_ngram_fallback_weight=float(
                _cfg_get(
                    "local_ngram_fallback_weight",
                    HybridModelAgent.LOCAL_NGRAM_FALLBACK_WEIGHT,
                )
            ),
            global_ngram_fallback_weight=float(
                _cfg_get(
                    "global_ngram_fallback_weight",
                    HybridModelAgent.GLOBAL_NGRAM_FALLBACK_WEIGHT,
                )
            ),
            lstm_train_steps=int(
                _cfg_get("lstm_train_steps", LstmWordModel.TRAIN_STEPS)
            ),
        )

    @app.post("/local/settings")
    async def update_settings(req: SettingsUpdateRequest) -> SettingsResponse:
        from .dl_module.hybrid_model import LstmWordModel, HybridModelAgent

        if req.auto_train is not None:
            _cfg_set("auto_train", bool(req.auto_train))
        if req.auto_share is not None:
            _cfg_set("auto_share", bool(req.auto_share))
        if req.max_concurrent_peer_sync is not None:
            val = max(1, min(50, int(req.max_concurrent_peer_sync)))
            _cfg_set("max_concurrent_peer_sync", val)
            app.state.share_semaphore = asyncio.Semaphore(val)
        if req.discovery_enable_lan is not None:
            _cfg_set("discovery_enable_lan", bool(req.discovery_enable_lan))
        if req.gossip_enabled is not None:
            _cfg_set("gossip_enabled", bool(req.gossip_enabled))
        if req.gossip_interval_s is not None:
            _cfg_set("gossip_interval_s", max(30, min(3600, int(req.gossip_interval_s))))

        # LSTM / DL tuning -- persist and apply live.
        if req.lstm_conf_threshold is not None:
            val = max(0.001, min(1.0, float(req.lstm_conf_threshold)))
            _cfg_set("lstm_conf_threshold", val)
            LstmWordModel.CONF_THRESHOLD = val
        if req.lstm_train_steps is not None:
            val = max(1, min(200, int(req.lstm_train_steps)))
            _cfg_set("lstm_train_steps", val)
            LstmWordModel.TRAIN_STEPS = val

        def _norm3(a: float, b: float, c: float) -> tuple[float, float, float]:
            t = a + b + c
            return (a / t, b / t, c / t) if t > 0 else (a, b, c)

        def _norm2(a: float, b: float) -> tuple[float, float]:
            t = a + b
            return (a / t, b / t) if t > 0 else (a, b)

        w_lstm = float(req.lstm_weight) if req.lstm_weight is not None else HybridModelAgent.LSTM_WEIGHT
        w_loc  = float(req.local_ngram_weight) if req.local_ngram_weight is not None else HybridModelAgent.LOCAL_NGRAM_WEIGHT
        w_glob = float(req.global_ngram_weight) if req.global_ngram_weight is not None else HybridModelAgent.GLOBAL_NGRAM_WEIGHT

        if any(x is not None for x in [req.lstm_weight, req.local_ngram_weight, req.global_ngram_weight]):
            w_lstm, w_loc, w_glob = _norm3(w_lstm, w_loc, w_glob)
            _cfg_set("lstm_weight", w_lstm)
            _cfg_set("local_ngram_weight", w_loc)
            _cfg_set("global_ngram_weight", w_glob)
            HybridModelAgent.LSTM_WEIGHT = w_lstm
            HybridModelAgent.LOCAL_NGRAM_WEIGHT = w_loc
            HybridModelAgent.GLOBAL_NGRAM_WEIGHT = w_glob

        w_fb_loc  = float(req.local_ngram_fallback_weight) if req.local_ngram_fallback_weight is not None else HybridModelAgent.LOCAL_NGRAM_FALLBACK_WEIGHT
        w_fb_glob = float(req.global_ngram_fallback_weight) if req.global_ngram_fallback_weight is not None else HybridModelAgent.GLOBAL_NGRAM_FALLBACK_WEIGHT

        if any(x is not None for x in [req.local_ngram_fallback_weight, req.global_ngram_fallback_weight]):
            w_fb_loc, w_fb_glob = _norm2(w_fb_loc, w_fb_glob)
            _cfg_set("local_ngram_fallback_weight", w_fb_loc)
            _cfg_set("global_ngram_fallback_weight", w_fb_glob)
            HybridModelAgent.LOCAL_NGRAM_FALLBACK_WEIGHT = w_fb_loc
            HybridModelAgent.GLOBAL_NGRAM_FALLBACK_WEIGHT = w_fb_glob

        return await get_settings()

    # -- LSTM / weights management --------------------------------------------

    @app.post("/local/model/upload_weights")
    async def upload_model_weights(
        model_id: str = Form(...), file: UploadFile = File(...)
    ) -> dict[str, Any]:
        """
        Accept a .json weights file and load it into the target model slot.

        Accepts two formats:
          1. Direct LstmWordModel payload: {arch, blob, train_steps, ...}
          2. Full HybridModelAgent federated payload: {lstm_state: {...}, ...}
        """
        if not file.filename or not file.filename.endswith(".json"):
            raise HTTPException(status_code=400, detail="Only .json weight files are accepted")
        raw = await file.read()
        if len(raw) > 50 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large (max 50 MB)")
        try:
            outer = json_loads(raw.decode("utf-8"))
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON")

        lstm_payload = (
            outer["lstm_state"]
            if "lstm_state" in outer and isinstance(outer["lstm_state"], dict)
            else outer
        )
        arch = lstm_payload.get("arch", "unknown")
        train_steps = lstm_payload.get("train_steps", 0)

        slot = registry.get(model_id)
        lstm = slot.model_agent.lstm

        if not lstm.available:
            return {
                "ok": False, "arch": arch, "train_steps": train_steps,
                "reason": (
                    "PyTorch not installed -- this node runs in n-gram-only mode. "
                    "Install torch to enable LSTM weight loading."
                ),
            }
        if lstm.model is None:
            return {
                "ok": False, "arch": arch, "train_steps": train_steps,
                "reason": "LSTM model not initialised -- trigger a prediction first.",
            }
        if lstm_payload.get("arch") != "word_lstm_v1":
            return {
                "ok": False, "arch": arch, "train_steps": train_steps,
                "reason": f"Unsupported arch '{arch}'. Expected 'word_lstm_v1'.",
            }
        if not lstm_payload.get("blob"):
            return {
                "ok": False, "arch": arch, "train_steps": train_steps,
                "reason": "Payload has no 'blob' key -- file may be corrupt or wrong format.",
            }

        ok = lstm.apply_peer_state(lstm_payload, mix=1.0)
        if ok:
            slot.model_agent.weight_version += 1
            slot.save()
            await hub.broadcast_json({"type": "weights_uploaded", "model_id": model_id})
            return {"ok": True, "arch": arch, "train_steps": train_steps}
        return {
            "ok": False, "arch": arch, "train_steps": train_steps,
            "reason": (
                "Weight tensor shapes do not match this model's vocabulary/architecture. "
                "Ensure the weights were exported from a model with the same vocab size."
            ),
        }

    @app.get("/local/model/torch_status")
    async def torch_status() -> dict[str, Any]:
        """Return PyTorch availability and the exact import error if any."""
        from .dl_module import hybrid_model as _hm
        import sys
        slots = list(registry._slots.values()) if hasattr(registry, "_slots") else []
        slot = slots[0] if slots else None
        lstm_available = slot.model_agent.lstm.available if slot else False
        model_ready = (slot.model_agent.lstm.model is not None) if slot else False
        return {
            "torch_importable": _hm._torch_import_error is None,
            "torch_import_error": _hm._torch_import_error,
            "lstm_available": lstm_available,
            "model_ready": model_ready,
            "python_executable": sys.executable,
            "python_version": sys.version,
        }

    @app.post("/local/model/reinit_lstm")
    async def reinit_lstm(model_id: str) -> dict[str, Any]:
        """Force a lazy torch re-import and LSTM reinitialisation for a model slot."""
        slot = registry.get(model_id)
        result = slot.model_agent.lstm.reinitialise()
        return {"ok": result["available"], **result}

    # -- Peer discovery -------------------------------------------------------

    @app.get("/local/peers/discovered")
    async def peers_discovered() -> dict[str, Any]:
        return {
            "ok": True,
            "peers": [p.to_dict() for p in disc.peers],
            "last_scan_ts": disc.last_scan_ts,
            "lan_enabled": disc.lan_enabled,
        }

    @app.post("/local/peers/scan")
    async def peers_scan() -> dict[str, Any]:
        enable_lan = bool(_cfg_get("discovery_enable_lan", False))
        found = await disc.scan(enable_lan=enable_lan)
        for p in found:
            reputation.upsert(p.url, peer_id=p.peer_id)
        await hub.broadcast_json({
            "type": "peers_discovered",
            "count": len(found),
            "peers": [p.to_dict() for p in found],
        })
        return {"ok": True, "peers": [p.to_dict() for p in found], "last_scan_ts": disc.last_scan_ts}

    @app.post("/local/peers/scan/settings")
    async def peers_scan_settings(enable_lan: bool = False) -> dict[str, Any]:
        _cfg_set("discovery_enable_lan", bool(enable_lan))
        return {"ok": True, "enable_lan": enable_lan}

    @app.get("/local/peers/reputation")
    async def peers_reputation() -> dict[str, Any]:
        return {"ok": True, "peers": reputation.all_peer_dicts()}

    # -- Federated endpoints --------------------------------------------------

    @app.post("/federated/delta")
    async def federated_receive_delta(body: dict[str, Any]) -> dict[str, Any]:
        """Receive an inbound peer delta and merge it immediately."""
        model_id = body.get("model_id", "default")
        delta = body.get("delta")
        if not delta:
            raise HTTPException(status_code=400, detail="missing delta")
        registry.get(model_id).apply_peer_delta(delta)
        await hub.broadcast_json({
            "type": "global_model_updated",
            "model_id": model_id,
            "deltas_merged": 1,
        })
        return {"ok": True}

    @app.get("/federated/weights")
    async def federated_serve_weights(model_id: str = "default") -> JSONResponse:
        """Serve the local hybrid payload for peer gossip pulls."""
        slot = registry.get(model_id)
        delta = slot.get_delta_payload(top_k=200)
        return JSONResponse(content=delta)

    @app.post("/local/share")
    async def share(req: ShareRequest) -> dict[str, Any]:
        slot = registry.get(req.model_id)
        payload = federated_sync.prepare_outbound(slot.model_agent, req.model_id)
        peers = req.peers if req.peers is not None else disc.get_active_urls()
        if not peers:
            return {"ok": True, "results": [], "note": "No peers discovered -- run a scan first"}
        sem = app.state.share_semaphore or asyncio.Semaphore(10)
        results = await federated_sync.share_delta(payload, peers, semaphore=sem)
        await hub.broadcast_json({
            "type": "share_complete",
            "model_id": req.model_id,
            "results": results,
        })
        return {"ok": True, "results": results}

    @app.post("/local/pull_global")
    async def pull_global(req: PullGlobalRequest) -> dict[str, Any]:
        peer_urls = disc.get_active_urls()
        if not peer_urls:
            raise HTTPException(status_code=503, detail="No discovered peers -- run a scan first")
        result = await gossip_engine.run_round(
            model_id=req.model_id,
            peer_urls=peer_urls,
            get_slot_fn=registry.get,
        )
        if not result.get("ok"):
            raise HTTPException(status_code=503, detail=f"Pull failed: {result.get('reason')}")
        await hub.broadcast_json({
            "type": "gossip_round_complete",
            "model_id": req.model_id,
            **result,
        })
        return {"ok": True, **result}

    @app.post("/gossip/round")
    async def gossip_round(req: GossipRoundRequest) -> dict[str, Any]:
        peer_urls = req.peers or disc.get_active_urls()
        if not peer_urls:
            raise HTTPException(status_code=503, detail="No discovered peers available.")
        result = await gossip_engine.run_round(
            model_id=req.model_id,
            peer_urls=peer_urls,
            get_slot_fn=registry.get,
        )
        await hub.broadcast_json({
            "type": "gossip_round_complete",
            "model_id": req.model_id,
            **result,
        })
        return {"ok": result.get("ok", False), **result}

    # -- Model registry -------------------------------------------------------

    @app.get("/local/models/list")
    async def models_list(db: SqliteDB = Depends(get_db)) -> dict[str, Any]:
        rows = db.query_all("SELECT * FROM models_registry ORDER BY created_at ASC;")
        result = []
        for r in rows:
            slot = registry.get(r["model_id"])
            entry = dict(r)
            entry["local_version"] = slot.local_version
            entry["global_version"] = slot.global_version
            entry["weight_version"] = slot.weight_version
            entry["engine"] = slot.engine
            entry["ngram_entries"] = slot.model_agent.local.count_entries()
            result.append(entry)
        return {"ok": True, "models": result}

    @app.post("/local/models/create")
    async def models_create(
        req: CreateModelRequest, db: SqliteDB = Depends(get_db)
    ) -> dict[str, Any]:
        new_id = str(uuid.uuid4())[:8]
        db.execute(
            "INSERT INTO models_registry(model_id, name, enabled, created_at, creation_type)"
            " VALUES(?, ?, 1, ?, ?);",
            (new_id, req.name, now_ts(), req.creation_type),
        )
        registry.get(new_id)
        await hub.broadcast_json({"type": "model_created", "model_id": new_id, "name": req.name})
        return {"ok": True, "model_id": new_id, "name": req.name}

    @app.post("/local/models/update")
    async def models_update(
        req: UpdateModelRequest, db: SqliteDB = Depends(get_db)
    ) -> dict[str, Any]:
        row = db.query_one("SELECT * FROM models_registry WHERE model_id=?;", (req.model_id,))
        if not row:
            raise HTTPException(status_code=404, detail="model not found")
        if req.enabled is not None:
            db.execute(
                "UPDATE models_registry SET enabled=? WHERE model_id=?;",
                (1 if req.enabled else 0, req.model_id),
            )
        if req.name is not None:
            db.execute(
                "UPDATE models_registry SET name=? WHERE model_id=?;",
                (req.name, req.model_id),
            )
        return {"ok": True}

    @app.post("/local/models/delete")
    async def models_delete(
        req: DeleteModelRequest, db: SqliteDB = Depends(get_db)
    ) -> dict[str, Any]:
        row = db.query_one("SELECT * FROM models_registry WHERE model_id=?;", (req.model_id,))
        if not row:
            raise HTTPException(status_code=404, detail="model not found")
        if req.model_id == "default":
            raise HTTPException(status_code=400, detail="cannot delete the default model")
        db.execute("DELETE FROM models_registry WHERE model_id=?;", (req.model_id,))
        return {"ok": True}

    # -- Personalisation ------------------------------------------------------

    @app.post("/local/personalization/observe")
    async def personalization_observe(req: ObserveTextRequest) -> dict[str, Any]:
        registry.get(req.model_id).observe_text(req.text)
        return {"ok": True}

    @app.post("/local/personalization/word/add")
    async def personalization_word_add(req: UserWordRequest) -> dict[str, Any]:
        registry.get(req.model_id).add_user_word(
            req.word, weight=req.weight, category=req.category
        )
        return {"ok": True, "word": req.word}

    @app.post("/local/personalization/word/remove")
    async def personalization_word_remove(req: UserWordRequest) -> dict[str, Any]:
        registry.get(req.model_id).personalization.user_dict.remove(req.word)
        return {"ok": True}

    @app.get("/local/personalization/words")
    async def personalization_words(model_id: str) -> dict[str, Any]:
        return {
            "ok": True,
            "words": registry.get(model_id).personalization.user_dict.to_list(),
        }

    @app.get("/local/personalization/suggestions")
    async def personalization_suggestions(model_id: str, k: int = 5) -> dict[str, Any]:
        return {
            "ok": True,
            "suggestions": registry.get(model_id).personalization.top_personal_suggestions(k=k),
        }

    # -- Prediction -----------------------------------------------------------

    @app.post("/local/predict")
    async def predict(req: PredictRequest) -> dict[str, Any]:
        slot = registry.get(req.model_id)
        suggestions, latency_ms = slot.predict(
            context_text=req.context_text,
            cursor_pos=req.cursor_pos,
            k=req.k,
            max_chars=req.max_chars,
        )
        return {
            "ok": True,
            "suggestions": [s.__dict__ for s in suggestions],
            "latency_ms": latency_ms,
            "model_versions": {
                "local": slot.local_version,
                "global": slot.global_version,
                "weights": slot.weight_version,
            },
            "engine": slot.engine,
        }

    # -- Sessions -------------------------------------------------------------

    @app.post("/local/session/start")
    async def session_start(
        req: SessionStartRequest, db: SqliteDB = Depends(get_db)
    ) -> dict[str, Any]:
        session_id = str(uuid.uuid4())
        db.execute(
            "INSERT INTO sessions(session_id, model_id, start_ts, end_ts, text_redacted, num_events)"
            " VALUES(?, ?, ?, NULL, '', 0);",
            (session_id, req.model_id, now_ts()),
        )
        return {"session_id": session_id}

    @app.post("/local/session/event")
    async def session_event(
        req: SessionEventRequest, db: SqliteDB = Depends(get_db)
    ) -> dict[str, Any]:
        db.execute(
            "INSERT INTO events(session_id, ts, type, payload_json) VALUES(?, ?, ?, ?);",
            (req.session_id, int(req.ts), req.type, json_dumps(req.payload)),
        )
        db.execute(
            "UPDATE sessions SET num_events=num_events+1 WHERE session_id=?;",
            (req.session_id,),
        )
        return {"ok": True}

    @app.post("/local/logs")
    async def logs(req: SessionEventRequest, db: SqliteDB = Depends(get_db)) -> dict[str, Any]:
        """Alias for /local/session/event used by the frontend logging path."""
        return await session_event(req, db)

    @app.post("/local/session/end")
    async def session_end(
        req: SessionEndRequest, db: SqliteDB = Depends(get_db)
    ) -> dict[str, Any]:
        redacted = redact_text(req.final_text or "")
        db.execute(
            "UPDATE sessions SET end_ts=?, text_redacted=? WHERE session_id=?;",
            (now_ts(), redacted, req.session_id),
        )
        db.execute(
            "INSERT INTO events(session_id, ts, type, payload_json) VALUES(?, ?, 'session_end', ?);",
            (req.session_id, now_ts(), json_dumps({"len": len(redacted)})),
        )
        row = db.query_one("SELECT model_id FROM sessions WHERE session_id=?;", (req.session_id,))
        if row:
            model_id = row["model_id"]
            slot = registry.get(model_id)
            slot.observe_text(redacted)
            auto_share = bool(_cfg_get("auto_share", False))
            peer_urls = disc.get_active_urls()
            agent = TrainingAgent(
                model_agent=slot.model_agent,
                cache_agent=slot.cache,
                federated_sync_agent=federated_sync,
                hub_broadcast_fn=hub.broadcast_json,
            )
            asyncio.create_task(
                agent.on_sentence_complete(
                    text=redacted,
                    model_id=model_id,
                    peer_urls=peer_urls,
                    auto_share=auto_share,
                )
            )
        return {"ok": True}

    @app.post("/local/train")
    async def train(req: TrainRequest) -> dict[str, Any]:
        rows = db.query_all(
            "SELECT text_redacted FROM sessions"
            " WHERE model_id=? AND end_ts IS NOT NULL"
            " ORDER BY end_ts DESC LIMIT 20;",
            (req.model_id,),
        )
        texts = [r["text_redacted"] for r in rows if r.get("text_redacted")]
        slot = registry.get(req.model_id)
        agent = TrainingAgent(
            model_agent=slot.model_agent,
            cache_agent=slot.cache,
            federated_sync_agent=federated_sync,
            hub_broadcast_fn=hub.broadcast_json,
        )
        words_learned = agent.manual_train(texts)
        await hub.broadcast_json({
            "type": "training_complete",
            "model_id": req.model_id,
            "local_version": slot.local_version,
            "global_version": slot.global_version,
            "words_learned": words_learned,
            "engine": slot.engine,
        })
        return {
            "ok": True,
            "words_learned": words_learned,
            "ngram_entries": slot.model_agent.local.count_entries(),
            "engine": slot.engine,
        }

    # -- Metrics / sessions ---------------------------------------------------

    @app.get("/local/metrics")
    async def metrics(model_id: str, db: SqliteDB = Depends(get_db)) -> dict[str, Any]:
        sessions = db.query_all(
            "SELECT text_redacted FROM sessions WHERE model_id=? AND end_ts IS NOT NULL;",
            (model_id,),
        )
        words = sum(len((s.get("text_redacted") or "").split()) for s in sessions)

        accepted = db.query_one(
            "SELECT COUNT(*) AS c FROM events WHERE type='suggest_accepted'"
            " AND session_id IN (SELECT session_id FROM sessions WHERE model_id=?);",
            (model_id,),
        )
        dismissed = db.query_one(
            "SELECT COUNT(*) AS c FROM events WHERE type='suggest_dismissed'"
            " AND session_id IN (SELECT session_id FROM sessions WHERE model_id=?);",
            (model_id,),
        )
        a = int(accepted["c"]) if accepted else 0
        d = int(dismissed["c"]) if dismissed else 0
        accept_rate = float(a / (a + d)) if (a + d) else 0.0

        lat_rows = db.query_all(
            "SELECT payload_json FROM events"
            " WHERE type IN ('suggest_accepted','suggest_dismissed')"
            " AND session_id IN (SELECT session_id FROM sessions WHERE model_id=?);",
            (model_id,),
        )
        latencies = []
        for r in lat_rows:
            try:
                latencies.append(int(json_loads(r["payload_json"]).get("latency_ms", 0)))
            except Exception:
                pass
        avg_latency = float(sum(latencies) / len(latencies)) if latencies else 0.0

        slot = registry.get(model_id)
        return {
            "ok": True,
            "model_id": model_id,
            "words_typed": words,
            "accepted": a,
            "dismissed": d,
            "accept_rate": accept_rate,
            "avg_latency_ms": avg_latency,
            "model_versions": {
                "local": slot.local_version,
                "global": slot.global_version,
                "weights": slot.weight_version,
            },
            "engine": slot.engine,
            "ngram_entries": slot.model_agent.local.count_entries(),
            "cache_stats": slot.cache.stats(),
            "accuracy_over_time": [],
        }

    @app.get("/local/sessions")
    async def sessions(
        model_id: str, limit: int = 50, db: SqliteDB = Depends(get_db)
    ) -> dict[str, Any]:
        rows = db.query_all(
            "SELECT session_id, start_ts, end_ts, num_events,"
            " LENGTH(text_redacted) AS text_len"
            " FROM sessions WHERE model_id=? ORDER BY start_ts DESC LIMIT ?;",
            (model_id, int(limit)),
        )
        return {"ok": True, "sessions": rows}

    @app.get("/local/sessions/texts")
    async def sessions_texts(
        model_id: str, limit: int = 30, db: SqliteDB = Depends(get_db)
    ) -> dict[str, Any]:
        rows = db.query_all(
            "SELECT text_redacted FROM sessions"
            " WHERE model_id=? AND end_ts IS NOT NULL AND text_redacted != ''"
            " ORDER BY end_ts DESC LIMIT ?;",
            (model_id, int(limit)),
        )
        return {"ok": True, "texts": [r["text_redacted"] for r in rows]}

    # -- N-gram persistence ---------------------------------------------------

    @app.get("/local/ngram/export")
    async def ngram_export(model_id: str = "default") -> JSONResponse:
        slot = registry.get(model_id)
        data = slot.model_agent.to_persistence_dict()
        return JSONResponse(content={"ok": True, "model_id": model_id, "data": data})

    @app.post("/local/ngram/save")
    async def ngram_save(model_id: str = "default") -> dict[str, Any]:
        registry.get(model_id).save()
        return {"ok": True, "model_id": model_id}

    return app
