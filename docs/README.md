# NWP v12 -- Hybrid LSTM + N-gram Federated Keyboard Language Model

## Project Overview

NWP v12 is a local-first, privacy-preserving next-word prediction server designed for keyboard applications. It combines a CPU-resident word-level LSTM neural network with a trie-based n-gram model, fused through a confidence-gated scoring blend. The server participates in a peer-to-peer federated learning network where lightweight model deltas are exchanged across discovered peers without any central coordinator.

The system is designed to run fully on-device. PyTorch is an optional dependency: if unavailable, the server falls back transparently to n-gram-only prediction without any code-path changes.

---

## Key Features

- Hybrid LSTM + trigram/bigram/unigram language model with confidence-gated score blending
- Graceful no-torch fallback: full operation in n-gram-only mode when PyTorch is absent
- Federated gossip learning: peer-to-peer delta exchange over HTTP, no central server
- Automatic peer discovery on localhost ports 8001-8020 and optionally on LAN /24 subnets
- Per-model personalisation: recency-boosted re-ranking and user-defined dictionary
- Session-level phrase cache for sub-millisecond repeated-phrase lookups
- PII redaction (email, URL, IP, numeric sequences) before any local storage or training
- SQLite WAL-mode persistent config store; all runtime settings are DB-backed
- Multi-model registry: named model slots, each with independent n-gram, LSTM, and personalisation state
- WebSocket push channel for real-time UI event notifications

---

## System Components

| Component | Location | Role |
|---|---|---|
| Entry point | `__main__.py` | CLI argument parsing, uvicorn launch |
| FastAPI app factory | `Server/app.py` | All HTTP routes, background loops, startup |
| Hybrid inference engine | `Server/dl_module/hybrid_model.py` | LSTM + n-gram confidence-gated prediction |
| N-gram model | `Server/dl_module/trie_model.py` | Trigram trie with backoff and federated merge |
| Session cache | `Server/dl_module/cache_agent.py` | In-memory phrase cache, recency scoring |
| Personalisation | `Server/dl_module/personalization.py` | Recency boost, user dictionary re-ranker |
| Model registry | `Server/model_registry.py` | ModelSlot lifecycle; full prediction pipeline |
| Gossip engine | `FL/gossip.py` | Peer reputation tracking and gossip rounds |
| Federated sync | `FL/federated_sync.py` | Outbound push and inbound delta queue |
| Training agent | `FL/training_agent.py` | Post-session local update and optional share |
| Peer discovery | `Server/peer_discovery.py` | Port-scan-based peer detection |
| SQLite layer | `Server/db.py` | Thread-safe WAL wrapper, schema init |
| Migrations | `Server/db_migrations.py` | Version-guarded incremental schema changes |
| WebSocket hub | `Server/ws.py` | Broadcast to all connected UI clients |
| Settings | `Server/settings.py` | Immutable runtime config dataclass |
| PII redaction | `Server/dl_module/redact.py` | Regex-based scrubber applied before storage |
| Hashing | `Server/dl_module/hashing.py` | SHA-256 utilities |
| Frontend | `Frontend/` | Static HTML/CSS/JS UI served by the server |
| Seed corpus | `Data/seed_corpus.txt` | Startup vocabulary for n-gram and LSTM |

---

## Execution Instructions

### Requirements

Python 3.10 or later is required. Install runtime dependencies:

```
pip install fastapi uvicorn[standard] httpx "pydantic>=2"
```

To enable the LSTM engine:

```
pip install torch numpy
```

### Running the Server

```
python -m nwp_v12 [--host HOST] [--port PORT] [--data-dir DIR] [--static-dir DIR]
```

| Argument | Default | Description |
|---|---|---|
| `--host` | `127.0.0.1` | Bind address |
| `--port` | `8001` | Listen port. Must be in range 8001-8020 |
| `--data-dir` | `<cwd>/data/trainer` | Directory for SQLite DB, model state, device ID |
| `--static-dir` | `<package>/Frontend` | Directory containing `index.html` |

The port range 8001-8020 is enforced at startup. Ports outside this range are rejected with an error. This range is also the auto-discovery scan range; all peer detection is strictly limited to these ports.

### Example

```
python -m nwp_v12 --port 8001 --data-dir ./runtime_data
```

The web UI is available at `http://127.0.0.1:8001/` once the server is running.

To run a second node on the same machine for federated testing:

```
python -m nwp_v12 --port 8002 --data-dir ./runtime_data_2
```

---

## Hybrid Inference Logic

### Overview

The inference path is implemented in `HybridModelAgent.predict()` (`Server/dl_module/hybrid_model.py`). It combines three scored sources: a local LSTM, a local n-gram model, and a global (peer-aggregated) n-gram model. The combination is controlled by a confidence gate evaluated on each prediction call.

### Prediction Modes

Before scoring begins, the system determines whether the cursor is mid-word. If the text before the cursor ends with a word character (`\w$`), the mode is `autocomplete` and only n-gram trie prefix search is used (the LSTM is not queried in autocomplete mode because it operates at word boundaries). Otherwise the mode is `next_word` and all three sources are queried.

### Confidence Gating

The LSTM returns a list of word predictions alongside a `max_confidence` value: the maximum softmax probability in its output distribution. This value is compared against `LstmWordModel.CONF_THRESHOLD` (default: `0.05`, configurable via the `NWP_LSTM_CONF_THRESHOLD` environment variable or the settings API at runtime).

If `lstm_conf >= CONF_THRESHOLD`, the LSTM result is considered reliable and the **confident branch** is used. If `lstm_conf < CONF_THRESHOLD`, the LSTM result is discarded and the **fallback branch** is used.

### Score Blending

Before blending, each source's raw scores are independently max-normalised to `[0, 1]` so no single source dominates due to output scale differences.

**Confident branch** (LSTM reliable):

```
score = LSTM_WEIGHT * lstm_score
      + LOCAL_NGRAM_WEIGHT * local_ngram_score
      + GLOBAL_NGRAM_WEIGHT * global_ngram_score
```

Default weights: `LSTM_WEIGHT = 0.50`, `LOCAL_NGRAM_WEIGHT = 0.20`, `GLOBAL_NGRAM_WEIGHT = 0.30`. These three values are normalised to sum to 1.0 by the settings update path.

**Fallback branch** (LSTM not confident or PyTorch unavailable):

```
score = LOCAL_NGRAM_FALLBACK_WEIGHT * local_ngram_score
      + GLOBAL_NGRAM_FALLBACK_WEIGHT * global_ngram_score
```

Default weights: `LOCAL_NGRAM_FALLBACK_WEIGHT = 0.40`, `GLOBAL_NGRAM_FALLBACK_WEIGHT = 0.60`. These two values are also normalised to sum to 1.0.

The dominant source by contribution is recorded as the `source` field in each returned suggestion (`"lstm"`, `"ngram"`, or `"peer-ngram"`).

### Role of Local vs Global Models

The **local n-gram** model is built from the user's own typed text. It is updated in real time after each session via `HybridModelAgent.observe()`, which records all n-grams from the redacted session text.

The **global n-gram** model is populated exclusively by incoming federated deltas from peer nodes. It is updated by `HybridModelAgent.update_global()` which performs a count-sum merge of peer n-gram tries. The global model is given a higher default fallback weight (0.60 vs 0.40) reflecting that peer data is expected to provide broader vocabulary coverage than the local-only signal.

The **LSTM** is fine-tuned locally on the user's session text and also accepts peer weight updates via L2-normalised blending (`apply_peer_state()` with `mix=0.35` by default from gossip, or `mix=1.0` for direct upload).

All weight values (`lstm_weight`, `local_ngram_weight`, `global_ngram_weight`, `local_ngram_fallback_weight`, `global_ngram_fallback_weight`, `lstm_conf_threshold`) are stored in the SQLite `config` table and applied live to class-level attributes when updated via `POST /local/settings`.

### Full Prediction Pipeline (ModelSlot level)

After `HybridModelAgent.predict()` returns raw suggestions, `ModelSlot.predict()` applies two additional layers:

1. **Cache blend**: `CacheAgent.lookup()` is called for next-word mode only. Cache hits receive a bonus of `0.25 * cache_score`. An additional recency bonus of `0.10 * recency_score` is added for all candidates from the cache's word-frequency window.

2. **Personalisation re-rank**: `PersonalizationLayer.rerank()` adds `ALPHA * recency_score` (0.15) and `BETA * user_dict_score` (0.20) to each candidate's blended score and re-sorts.

Results are filtered to `max_chars` length (default 24, max 64) before the top-k are returned.

---

## High-level Data Flow

```
User types text
    |
    v
POST /local/predict
    |
    v
ModelSlot.predict()
    |--> CacheAgent.lookup()         (sub-ms phrase cache)
    |--> HybridModelAgent.predict()
    |       |--> LstmWordModel.predict()   (if mode=next_word and torch available)
    |       |--> NgramModel.predict()      (local n-gram, trigram/bigram/unigram backoff)
    |       |--> NgramModel.predict()      (global n-gram from peer merges)
    |       |--> confidence gate check
    |       |--> score blend (confident or fallback branch)
    |--> personalisation re-rank
    |
    v
JSON response: suggestions[], latency_ms, engine, model_versions

Session ends (POST /local/session/end)
    |
    v
redact_text()  -->  observe (n-gram + LSTM fine-tune)  -->  cache update
    -->  flush peer delta queue  -->  optional background share to peers
```

---

## Notes and Limitations (from code evidence)

- The `weight_version` tracker is incremented on LSTM weight uploads but the `_flush_loop` does not increment it. There is no automatic version sync on gossip-merged LSTM weights.
- The `peer_reputation` table created by migration 2 is never written to by the server; `PeerReputation` is entirely in-memory and does not survive restarts.
- The `personalization_state` table created by migration 3 is never written to by the server; `PersonalizationLayer` state is persisted to `pers_<model_id>.json` files, not the DB.
- The `ngram_sync_log` table created by migration 4 is never written to by any code path.
- The `weights` table created by `_create_core_tables` is never written to by any code path.
- The `federated/delta` endpoint calls `apply_peer_delta` immediately (bypassing `FederatedSyncAgent._pending_deltas`), while `_flush_loop` processes the queue path. The two inbound paths are parallel and the queue path is only exercised if `receive_delta` is called directly, which no current code path does.
- LAN scanning probes every host in the local /24 subnet (up to 254 hosts × 20 ports = 5,080 concurrent HTTP requests). There is no rate limiting or chunking on this scan.
- `ModelSlot.weight_version` is defined as a property that reads `self.model_agent.weight_version`, but `HybridModelAgent.weight_version` is only incremented in `apply_federated_payload` (when LSTM merge succeeds) and in `upload_weights` (direct upload path). Gossip-sourced LSTM merges that succeed increment the counter; those that fail do not.
- Session texts are stored as `text_redacted` but the redaction is applied only at session end, not to events logged mid-session.
- The version string in `app.py` reads `"version": "11.0"` while `pyproject.toml` declares `version = "12.0.0"`.
