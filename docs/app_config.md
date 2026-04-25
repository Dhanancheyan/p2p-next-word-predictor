# app_config.md -- NWP v12 Configuration Reference

## Configuration System

All runtime configuration is stored in the SQLite `config` table as JSON-encoded key-value pairs. This is a non-standard design: there is no configuration file on disk. Settings are read from the database on every relevant request and written back immediately on update. Selected class-level attributes on `LstmWordModel` and `HybridModelAgent` are also mutated in-process by `update_settings()` to avoid requiring a restart.

The `config` table schema:

```sql
CREATE TABLE config (
    key        TEXT PRIMARY KEY,
    value_json TEXT NOT NULL
);
```

Two immutable settings are supplied at process startup via CLI arguments and are not stored in the database:

| Setting | Source | Description |
|---|---|---|
| `host` | CLI `--host` | Bind address. Default `127.0.0.1` |
| `port` | CLI `--port` | Listen port. Must be 8001-8020. Default `8001` |
| `data_dir` | CLI `--data-dir` | Root data directory. Default `<cwd>/data/trainer` |
| `static_dir` | CLI `--static-dir` | Frontend directory. Default `<package>/Frontend` |
| `device_id` | `device_id.txt` in data_dir | Stable UUID generated on first run, persisted as plain text |

---

## DB-Backed Config Keys

All keys below are read via `_cfg_get(key, default)` and written via `_cfg_set(key, value)`. The API to read/write them is `GET /local/settings` and `POST /local/settings`.

### Peer Discovery

| Key | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `discovery_enable_lan` | bool | `false` | — | When true, scan all hosts on the local /24 subnet in addition to localhost. LAN scan probes up to 254 hosts × 20 ports concurrently. |

### Federated Learning

| Key | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `auto_share` | bool | `false` | — | Automatically push a federated delta to discovered peers at the end of each session. |
| `gossip_enabled` | bool | `true` | — | Enable the background gossip loop. |
| `gossip_interval_s` | int | `300` | min 30, max 3600 | Interval between gossip rounds in seconds. |
| `max_concurrent_peer_sync` | int | `10` | min 1, max 50 | Maximum simultaneous outbound peer connections. Applied as an asyncio Semaphore. Changes take effect immediately. |

### Training

| Key | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `auto_train` | bool | `true` | — | Flag stored in DB. Read by `GET /local/settings`. Not currently used to gate any training code path in the server. |
| `lstm_train_steps` | int | `LstmWordModel.TRAIN_STEPS` = `10` | min 1, max 200 | Maximum gradient steps per training call. Applied live to `LstmWordModel.TRAIN_STEPS`. |

### Hybrid Inference Weights

These keys control the confidence-gated score blend. On update, the server normalises each group so weights sum to 1.0 before storing.

**Confident branch** (LSTM max-softmax >= `lstm_conf_threshold`):

| Key | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `lstm_conf_threshold` | float | `LstmWordModel.CONF_THRESHOLD` = `0.05` (or `NWP_LSTM_CONF_THRESHOLD` env var) | 0.001 – 1.0 | Minimum LSTM max-softmax probability to activate the LSTM branch. Applied live to `LstmWordModel.CONF_THRESHOLD`. |
| `lstm_weight` | float | `0.50` | Normalised with local/global | Weight of LSTM predictions in the confident branch. |
| `local_ngram_weight` | float | `0.20` | Normalised with lstm/global | Weight of local n-gram predictions in the confident branch. |
| `global_ngram_weight` | float | `0.30` | Normalised with lstm/local | Weight of global (peer) n-gram predictions in the confident branch. |

**Fallback branch** (LSTM below threshold or PyTorch unavailable):

| Key | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `local_ngram_fallback_weight` | float | `0.40` | Normalised with global | Weight of local n-gram in the fallback branch. |
| `global_ngram_fallback_weight` | float | `0.60` | Normalised with local | Weight of global n-gram in the fallback branch. |

### DB Version (internal)

| Key | Type | Description |
|---|---|---|
| `db_version` | int | Current migration version. Managed by `db_migrations.py`. Do not write manually. |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `NWP_LSTM_CONF_THRESHOLD` | `0.05` | Sets `LstmWordModel.CONF_THRESHOLD` at class definition time. Can be overridden at runtime via the settings API. |

---

## Federated Payload Schemas

### Outbound delta payload (`FederatedSyncAgent.prepare_outbound`)

Sent to `POST /federated/delta` on each peer.

```json
{
  "device_id": "<string>",
  "model_id": "<string>",
  "ts": <unix_timestamp_int>,
  "delta": {
    "version": 2,
    "kind": "hybrid_lstm_ngram",
    "engine": "lstm+ngram | ngram-fallback",
    "ngram": {
      "version": 1,
      "max_ngram": 3,
      "unigram": { "<word>": <count_int>, ... },
      "total_unigrams": <int>,
      "ngrams": {
        "<ctx_word1>": { "<next_word>": <count_int>, ... },
        "<ctx_word1> <ctx_word2>": { "<next_word>": <count_int>, ... }
      }
    },
    "lstm_state": {
      "format": "torch_state_gzip_b64",
      "arch": "word_lstm_v1",
      "train_steps": <int>,
      "blob": "<base64-encoded gzip PyTorch state_dict>"
    }
  }
}
```

`lstm_state` is `null` when PyTorch is unavailable.

### Inbound weights request (`GET /federated/weights`)

Response body is the `delta` dict from the outbound payload above (without the outer envelope).

### LSTM weight file format (for `POST /local/model/upload_weights`)

Accepts either:

1. Direct LSTM payload:
```json
{
  "arch": "word_lstm_v1",
  "train_steps": <int>,
  "blob": "<base64-encoded gzip PyTorch state_dict>"
}
```

2. Full hybrid payload (outer envelope):
```json
{
  "lstm_state": {
    "arch": "word_lstm_v1",
    "train_steps": <int>,
    "blob": "<base64-encoded gzip PyTorch state_dict>"
  },
  ...
}
```

The server detects format by presence of the `lstm_state` key. Maximum file size: 50 MB.

---

## Model Parameters

### `_WordLSTM` architecture

| Parameter | Value |
|---|---|
| Type | Word-level LSTM |
| Layers | Embedding → LSTM (2-layer not explicit, 1 in code) → Linear head |
| `embed_dim` | 96 |
| `hidden_dim` | 160 |
| `vocab_size` | `len(word_to_id)` — built from seed corpus, capped at 2500 |
| Optimizer | AdamW, `lr=0.003` |
| Gradient clipping | `clip_grad_norm_` max norm 1.0 |
| Max sequence length | 16 tokens |
| Training device | CPU only |

### `NgramModel` parameters

| Parameter | Value |
|---|---|
| Max n-gram order | 3 (trigram) |
| Vocabulary cap | 1000 words |
| Smoothing | Laplace, constant 0.5 |
| Eviction policy | Least-frequent word (ties broken alphabetically) |
| Backoff | Trigram → bigram → unigram |

### `CacheAgent` parameters

| Parameter | Value |
|---|---|
| Phrase cache capacity | 512 LRU keys |
| Word window capacity | 256 entries |
| Cache weight in blend | 0.25 |
| Recency weight in blend | 0.10 |

### `PersonalizationLayer` parameters

| Parameter | Value |
|---|---|
| `ALPHA` (recency boost) | 0.15 |
| `BETA` (user dict boost) | 0.20 |
| User dict weight clamp | 0.5 – 2.0 (enforced by caller; not enforced in `UserDictEntry`) |
| Cache capacity | 256 words |

---

## SQLite Database Tables

| Table | Purpose |
|---|---|
| `config` | Key-value store for all runtime settings and `db_version` |
| `models_registry` | Named model slots (`model_id`, `name`, `enabled`, `created_at`, `creation_type`) |
| `sessions` | One row per typing session; stores redacted text and event count |
| `events` | Per-session interaction events (`suggest_accepted`, `suggest_dismissed`, etc.) |
| `weights` | Metadata for weight files (created in schema, never written by server code) |
| `peer_reputation` | Migration 2; never written by server code (state is in-memory only) |
| `personalization_state` | Migration 3; never written by server code (state is in JSON files) |
| `ngram_sync_log` | Migration 4; never written by server code |

**Indexes:**

| Index | Table | Columns |
|---|---|---|
| `idx_sessions_model_start` | `sessions` | `(model_id, start_ts)` |
| `idx_events_session_ts` | `events` | `(session_id, ts)` |
| `idx_weights_model_kind_version` | `weights` | `(model_id, kind, version)` |
