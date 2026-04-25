# API_REFERENCE.md -- NWP v12 REST and WebSocket Reference

All endpoints are served on the configured host and port (default `http://127.0.0.1:8001`).

---

## Local Operations

### `GET /`

Serves `Frontend/index.html`. Returns HTTP 500 if the file is missing.

---

### `GET /static/{path}`

Serves static files from the Frontend directory. Path traversal is blocked.

---

### `GET /health`

Peer probe endpoint. Used by `PeerDiscovery` to identify running instances.

Response:
```json
{
  "ok": true,
  "peer_id": "<device_id>",
  "version": "11.0",
  "engine": "hybrid-lstm-v11"
}
```

Internal mapping: reads `settings.device_id`.

---

### `GET /local/status`

Returns device ID and number of discovered peers.

Response:
```json
{
  "ok": true,
  "device_id": "<string>",
  "discovered_peers": <int>
}
```

---

### `GET /local/settings`

Returns all configurable parameters with current values (from DB + class defaults).

Response: `SettingsResponse` — see `app_config.md` for all fields.

---

### `POST /local/settings`

Updates one or more configurable parameters. Partial updates are supported. Weight groups are normalised to sum to 1.0 on update. Changes are persisted to DB and applied live.

Request body (all fields optional):
```json
{
  "auto_train": bool,
  "auto_share": bool,
  "max_concurrent_peer_sync": int,
  "discovery_enable_lan": bool,
  "gossip_enabled": bool,
  "gossip_interval_s": int,
  "lstm_conf_threshold": float,
  "lstm_weight": float,
  "local_ngram_weight": float,
  "global_ngram_weight": float,
  "local_ngram_fallback_weight": float,
  "global_ngram_fallback_weight": float,
  "lstm_train_steps": int
}
```

Response: same as `GET /local/settings` reflecting updated values.

---

### `POST /local/predict`

Run next-word prediction for a given context.

Request:
```json
{
  "model_id": "<string>",
  "context_text": "<string>",
  "cursor_pos": <int>,
  "k": <int 1-5, default 5>,
  "max_chars": <int 1-64, default 24>
}
```

Response:
```json
{
  "ok": true,
  "suggestions": [
    {
      "text": "<string>",
      "score": <float>,
      "source": "lstm | ngram | peer-ngram | cache | hybrid",
      "mode": "next_word | autocomplete",
      "partial_word": "<string>",
      "lstm_conf": <float>
    }
  ],
  "latency_ms": <float>,
  "model_versions": {
    "local": <int>,
    "global": <int>,
    "weights": <int>
  },
  "engine": "lstm+ngram | ngram-fallback"
}
```

Internal call chain: `ModelSlot.predict()` → `CacheAgent.lookup()` + `HybridModelAgent.predict()` → `PersonalizationLayer.rerank()`.

---

## Sessions and Training

### `POST /local/session/start`

Start a new session. Returns a UUID session ID.

Request:
```json
{ "model_id": "<string>" }
```

Response:
```json
{ "session_id": "<uuid>" }
```

---

### `POST /local/session/event`

Log a session event (e.g., `suggest_accepted`, `suggest_dismissed`).

Request:
```json
{
  "session_id": "<string min 8>",
  "ts": <int unix timestamp, auto if omitted>,
  "type": "<string>",
  "payload": { ... }
}
```

Response: `{ "ok": true }`

---

### `POST /local/logs`

Alias for `POST /local/session/event`. Used by the frontend logging path.

---

### `POST /local/session/end`

End a session. Triggers: PII redaction, n-gram + LSTM training, cache update, and optional federated share.

Request:
```json
{
  "session_id": "<string min 8>",
  "final_text": "<string>"
}
```

Response: `{ "ok": true }`

Internal: `redact_text()` → DB update → `ModelSlot.observe_text()` → `asyncio.create_task(TrainingAgent.on_sentence_complete(...))`.

---

### `POST /local/train`

Manually trigger training from the 20 most recent completed sessions.

Request:
```json
{
  "model_id": "<string>",
  "reason": "manual | session_end | scheduled"
}
```

Response:
```json
{
  "ok": true,
  "words_learned": <int>,
  "ngram_entries": <int>,
  "engine": "<string>"
}
```

Internal: queries `sessions` table → `TrainingAgent.manual_train(texts)`.

---

### `GET /local/sessions`

List recent sessions for a model.

Query params: `model_id=<string>`, `limit=<int default 50>`

Response:
```json
{
  "ok": true,
  "sessions": [
    {
      "session_id": "<string>",
      "start_ts": <int>,
      "end_ts": <int or null>,
      "num_events": <int>,
      "text_len": <int>
    }
  ]
}
```

---

### `GET /local/sessions/texts`

Return redacted text from recent completed sessions.

Query params: `model_id=<string>`, `limit=<int default 30>`

Response:
```json
{ "ok": true, "texts": ["<string>", ...] }
```

---

## Model Registry

### `GET /local/models/list`

List all registered model slots with runtime stats.

Response:
```json
{
  "ok": true,
  "models": [
    {
      "model_id": "<string>",
      "name": "<string>",
      "enabled": <0|1>,
      "created_at": <int>,
      "creation_type": "<string>",
      "local_version": <int>,
      "global_version": <int>,
      "weight_version": <int>,
      "engine": "<string>",
      "ngram_entries": <int>
    }
  ]
}
```

---

### `POST /local/models/create`

Create a new named model slot.

Request:
```json
{
  "name": "<string max 80>",
  "creation_type": "default | blank | clone",
  "source_model_id": "<string or null>"
}
```

Response:
```json
{ "ok": true, "model_id": "<8-char uuid prefix>", "name": "<string>" }
```

Note: `source_model_id` is accepted in the request schema but is not used in the model creation logic.

---

### `POST /local/models/update`

Rename or toggle a model slot.

Request:
```json
{
  "model_id": "<string>",
  "name": "<string or null>",
  "enabled": <bool or null>
}
```

Response: `{ "ok": true }`

---

### `POST /local/models/delete`

Delete a model slot. The `default` model cannot be deleted.

Request:
```json
{ "model_id": "<string>" }
```

Response: `{ "ok": true }` or HTTP 400/404.

---

## LSTM / Weight Management

### `POST /local/model/upload_weights`

Upload an LSTM weights file. Accepts two JSON formats (see `app_config.md`).

Request: multipart/form-data with fields:
- `model_id`: string
- `file`: `.json` file (max 50 MB)

Response:
```json
{
  "ok": true,
  "arch": "word_lstm_v1",
  "train_steps": <int>
}
```

On failure, `ok` is `false` with a `reason` string. LSTM must be available and model must have been initialised with at least one prediction.

---

### `GET /local/model/torch_status`

Returns PyTorch availability diagnostics.

Response:
```json
{
  "torch_importable": <bool>,
  "torch_import_error": "<string or null>",
  "lstm_available": <bool>,
  "model_ready": <bool>,
  "python_executable": "<string>",
  "python_version": "<string>"
}
```

---

### `POST /local/model/reinit_lstm`

Force torch re-import and LSTM reinitialisation for a model slot.

Query param: `model_id=<string>`

Response:
```json
{
  "ok": <bool>,
  "available": <bool>,
  "torch_error": "<string or null>",
  "model_ready": <bool>
}
```

---

## Federated / Distributed

### `POST /federated/delta`

Receive an inbound peer delta and merge it immediately into the target model's global n-gram + LSTM.

Request body:
```json
{
  "model_id": "<string>",
  "delta": { <hybrid payload, see app_config.md> }
}
```

Response: `{ "ok": true }`

Internal: `registry.get(model_id).apply_peer_delta(delta)` → `broadcast global_model_updated`.

---

### `GET /federated/weights`

Serve the local hybrid payload for peer gossip pulls.

Query param: `model_id=<string default "default">`

Response: hybrid payload dict (see `app_config.md` for schema).

Internal: `ModelSlot.get_delta_payload(top_k=200)`.

---

### `POST /local/share`

Manually push the local model delta to peers.

Request:
```json
{
  "model_id": "<string>",
  "peers": ["<url>", ...] | null,
  "mode": "manual | scheduled"
}
```

If `peers` is null, uses currently discovered peers. If no peers found, returns note.

Response:
```json
{
  "ok": true,
  "results": [
    { "peer": "<url>", "ok": <bool>, "error": "<string or absent>" }
  ]
}
```

---

### `POST /local/pull_global`

Manually trigger a gossip round to pull a peer delta.

Request:
```json
{ "model_id": "<string>" }
```

Response:
```json
{
  "ok": true,
  "round_id": "<string>",
  "peer_url": "<string>",
  "latency_ms": <float>
}
```

HTTP 503 if no peers discovered. HTTP 503 if all peers fail.

---

### `POST /gossip/round`

Trigger a gossip round with explicit or auto-discovered peers.

Request:
```json
{
  "model_id": "<string>",
  "peers": ["<url>", ...] | null
}
```

Response:
```json
{
  "ok": <bool>,
  "round_id": "<string>",
  "peer_url": "<string or absent>",
  "reason": "<string or absent>"
}
```

---

## Peer Discovery

### `GET /local/peers/discovered`

Returns cached discovered peers from the last scan.

Response:
```json
{
  "ok": true,
  "peers": [
    {
      "url": "<string>",
      "peer_id": "<string>",
      "latency_ms": <float>,
      "status": "ok | error",
      "meta": { ... },
      "last_seen": <int>
    }
  ],
  "last_scan_ts": <int>,
  "lan_enabled": <bool>
}
```

---

### `POST /local/peers/scan`

Run a fresh peer scan immediately.

Response: same shape as `GET /local/peers/discovered`.

---

### `POST /local/peers/scan/settings`

Enable or disable LAN scanning.

Query param: `enable_lan=<bool>`

Response: `{ "ok": true, "enable_lan": <bool> }`

---

### `GET /local/peers/reputation`

Return all peer reputation records.

Response:
```json
{
  "ok": true,
  "peers": [
    {
      "url": "<string>",
      "peer_id": "<string>",
      "score": <float>,
      "successes": <int>,
      "failures": <int>,
      "avg_latency_ms": <float>,
      "last_seen": <int>,
      "model_version": <int>
    }
  ]
}
```

Note: reputation is in-memory only; data is lost on server restart.

---

## Personalisation

### `POST /local/personalization/observe`

Feed text into the personalisation recency cache.

Request:
```json
{ "model_id": "<string>", "text": "<string>" }
```

Response: `{ "ok": true }`

---

### `POST /local/personalization/word/add`

Add a word to the user dictionary for a model.

Request:
```json
{
  "model_id": "<string>",
  "word": "<string min 1>",
  "weight": <float default 1.0>,
  "category": "<string default 'custom'>"
}
```

Response: `{ "ok": true, "word": "<string>" }`

---

### `POST /local/personalization/word/remove`

Remove a word from the user dictionary.

Request: same shape as add (only `model_id` and `word` are used).

Response: `{ "ok": true }`

---

### `GET /local/personalization/words`

List all user dictionary entries for a model.

Query param: `model_id=<string>`

Response:
```json
{
  "ok": true,
  "words": [
    {
      "word": "<string>",
      "weight": <float>,
      "category": "<string>",
      "created_at": <float unix timestamp>,
      "use_count": <int>
    }
  ]
}
```

---

### `GET /local/personalization/suggestions`

Return the most frequently used words from the recency cache.

Query params: `model_id=<string>`, `k=<int default 5>`

Response:
```json
{ "ok": true, "suggestions": ["<word>", ...] }
```

---

## Metrics and Logs

### `GET /local/metrics`

Return usage and performance metrics for a model.

Query param: `model_id=<string>`

Response:
```json
{
  "ok": true,
  "model_id": "<string>",
  "words_typed": <int>,
  "accepted": <int>,
  "dismissed": <int>,
  "accept_rate": <float 0.0-1.0>,
  "avg_latency_ms": <float>,
  "model_versions": { "local": <int>, "global": <int>, "weights": <int> },
  "engine": "<string>",
  "ngram_entries": <int>,
  "cache_stats": {
    "phrase_keys": <int>,
    "word_window_size": <int>,
    "unique_words": <int>
  },
  "accuracy_over_time": []
}
```

Note: `accuracy_over_time` is always an empty list (not implemented).

---

## N-gram Persistence

### `GET /local/ngram/export`

Export the full model state as JSON (local + global n-gram + versions).

Query param: `model_id=<string default "default">`

Response:
```json
{
  "ok": true,
  "model_id": "<string>",
  "data": { <to_persistence_dict output> }
}
```

---

### `POST /local/ngram/save`

Persist the current model state to disk.

Query param: `model_id=<string default "default">`

Response: `{ "ok": true, "model_id": "<string>" }`

---

## WebSocket

### `WS /ws/local`

Server-to-client event push channel. The server broadcasts JSON events for all significant state changes.

Event types:

| Type | When emitted | Payload fields |
|---|---|---|
| `peers_discovered` | After peer scan | `count`, `peers[]` |
| `gossip_round_complete` | After gossip round | `model_id`, `round_id`, `peer_url`, `latency_ms` |
| `global_model_updated` | After delta merge | `model_id`, `deltas_merged` |
| `share_complete` | After outbound share | `model_id`, `results[]` or `peers_ok`, `peers_total` |
| `training_started` | On session end (async) | `model_id` |
| `training_complete` | After training (async) | `model_id`, `local_version`, `global_version`, `words_learned`, `engine` |
| `weights_uploaded` | After weight file upload | `model_id` |
| `model_created` | After model creation | `model_id`, `name` |

The client sends arbitrary text to keep the connection alive; messages are discarded by the server.
