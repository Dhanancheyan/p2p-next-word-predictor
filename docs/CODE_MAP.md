# CODE_MAP.md -- NWP v12 Structural Map

## Entry Point Trace

```
python -m nwp_v12
    |
    nwp_v12/__main__.py :: main()
        |-- argparse (--host, --port, --data-dir, --static-dir)
        |-- Server/settings.py :: build_settings()
        |       |-- _load_or_create_device_id()
        |       |-- returns TrainerSettings (frozen dataclass)
        |
        |-- Server/app.py :: create_app(settings, static_dir)
        |       |-- returns FastAPI instance with all routes registered
        |
        |-- uvicorn.run(app, host, port)
```

---

## File Index

| File | Package | Role |
|---|---|---|
| `__main__.py` | `nwp_v12` | CLI entry point; argument parsing; uvicorn launch |
| `Server/__init__.py` | `nwp_v12.Server` | Package docstring only |
| `Server/app.py` | `nwp_v12.Server` | FastAPI app factory; all REST/WebSocket routes; 3 background asyncio loops |
| `Server/settings.py` | `nwp_v12.Server` | `TrainerSettings` frozen dataclass; device ID persistence |
| `Server/db.py` | `nwp_v12.Server` | `SqliteDB` thread-safe wrapper; WAL mode; schema creation; `init_db()` |
| `Server/db_migrations.py` | `nwp_v12.Server` | Version-guarded migrations (v1-v4); `run_migrations()` |
| `Server/model_registry.py` | `nwp_v12.Server` | `ModelSlot` (full prediction pipeline); `ModelRegistry` (lazy-loading cache) |
| `Server/peer_discovery.py` | `nwp_v12.Server` | `PeerDiscovery` (port-scan based); `DiscoveredPeer` dataclass |
| `Server/ws.py` | `nwp_v12.Server` | `WebSocketHub` (broadcast to all connected UI clients) |
| `Server/dl_module/__init__.py` | `nwp_v12.Server.dl_module` | Package docstring only |
| `Server/dl_module/hybrid_model.py` | `nwp_v12.Server.dl_module` | `LstmWordModel`; `HybridModelAgent`; `_WordLSTM` (nn.Module); confidence-gated blend |
| `Server/dl_module/trie_model.py` | `nwp_v12.Server.dl_module` | `NgramModel` (trie-based trigram LM); `TrieNode`; `build_seed_model()`; `tokenize()` |
| `Server/dl_module/cache_agent.py` | `nwp_v12.Server.dl_module` | `CacheAgent` (LRU phrase cache + recency window) |
| `Server/dl_module/personalization.py` | `nwp_v12.Server.dl_module` | `PersonalizationLayer`; `RecentWordCache`; `UserDictionary`; `UserDictEntry` |
| `Server/dl_module/redact.py` | `nwp_v12.Server.dl_module` | `redact_text()` (email, URL, IP, NUM patterns) |
| `Server/dl_module/hashing.py` | `nwp_v12.Server.dl_module` | `sha256_hex()`; `canonical_json_dumps()`; `sha256_hex_of_canonical_json()` |
| `FL/__init__.py` | `nwp_v12.FL` | Package docstring only |
| `FL/gossip.py` | `nwp_v12.FL` | `GossipEngine`; `PeerReputation`; `PeerRecord` dataclass |
| `FL/federated_sync.py` | `nwp_v12.FL` | `FederatedSyncAgent` (outbound push, inbound queue, flush) |
| `FL/training_agent.py` | `nwp_v12.FL` | `TrainingAgent` (post-session update pipeline) |
| `Data/seed_corpus.txt` | -- | Startup vocabulary (one sentence per line) |
| `Data/lstm_weights_placeholder.json` | -- | Documents LSTM weight file format |
| `Data/README.md` | -- | Data directory documentation |
| `Frontend/index.html` | -- | Web UI entry point |
| `Frontend/app.js` | -- | Frontend JavaScript application |
| `Frontend/styles.css` | -- | Frontend stylesheet |
| `requirements.txt` | -- | Pip dependency list with optional torch |
| `pyproject.toml` | -- | Build metadata; setuptools configuration |

---

## Module-by-Module Breakdown

### `Server/app.py`

**Functions (all closures inside `create_app`):**

| Name | Type | Purpose |
|---|---|---|
| `create_app(settings, static_dir)` | Factory | Constructs and returns the configured FastAPI app |
| `get_db()` | Dependency | Returns the shared `SqliteDB` instance |
| `_cfg_get(key, default)` | Helper | Reads a value from the `config` table |
| `_cfg_set(key, value)` | Helper | Writes a value to the `config` table (upsert) |
| `_ensure_defaults()` | Startup | Creates the `default` model row if absent |
| `_load_active_models()` | Startup | Pre-warms all enabled model slots |
| `_auto_discovery_loop()` | Async task | Scans for peers every 60 s; broadcasts results |
| `_gossip_loop()` | Async task | Runs gossip FL rounds at `gossip_interval_s` |
| `_flush_loop()` | Async task | Drains inbound delta queue every 30 s |
| `_startup()` | Event handler | Calls startup helpers; creates background tasks |
| `index()` | Route | `GET /` -- serves `index.html` |
| `static_files(path)` | Route | `GET /static/{path}` -- serves Frontend assets |
| `ws_local(ws)` | Route | `WS /ws/local` -- connects client to WebSocket hub |
| `health()` | Route | `GET /health` -- peer probe endpoint |
| `status()` | Route | `GET /local/status` |
| `get_settings()` | Route | `GET /local/settings` |
| `update_settings(req)` | Route | `POST /local/settings` -- updates DB and live class attributes |
| `upload_model_weights(model_id, file)` | Route | `POST /local/model/upload_weights` |
| `torch_status()` | Route | `GET /local/model/torch_status` |
| `reinit_lstm(model_id)` | Route | `POST /local/model/reinit_lstm` |
| `peers_discovered()` | Route | `GET /local/peers/discovered` |
| `peers_scan()` | Route | `POST /local/peers/scan` |
| `peers_scan_settings(enable_lan)` | Route | `POST /local/peers/scan/settings` |
| `peers_reputation()` | Route | `GET /local/peers/reputation` |
| `federated_receive_delta(body)` | Route | `POST /federated/delta` |
| `federated_serve_weights(model_id)` | Route | `GET /federated/weights` |
| `share(req)` | Route | `POST /local/share` |
| `pull_global(req)` | Route | `POST /local/pull_global` |
| `gossip_round(req)` | Route | `POST /gossip/round` |
| `models_list(db)` | Route | `GET /local/models/list` |
| `models_create(req, db)` | Route | `POST /local/models/create` |
| `models_update(req, db)` | Route | `POST /local/models/update` |
| `models_delete(req, db)` | Route | `POST /local/models/delete` |
| `personalization_observe(req)` | Route | `POST /local/personalization/observe` |
| `personalization_word_add(req)` | Route | `POST /local/personalization/word/add` |
| `personalization_word_remove(req)` | Route | `POST /local/personalization/word/remove` |
| `personalization_words(model_id)` | Route | `GET /local/personalization/words` |
| `personalization_suggestions(model_id, k)` | Route | `GET /local/personalization/suggestions` |
| `predict(req)` | Route | `POST /local/predict` |
| `session_start(req, db)` | Route | `POST /local/session/start` |
| `session_event(req, db)` | Route | `POST /local/session/event` |
| `logs(req, db)` | Route | `POST /local/logs` (alias for session/event) |
| `session_end(req, db)` | Route | `POST /local/session/end` |
| `train(req)` | Route | `POST /local/train` |
| `metrics(model_id, db)` | Route | `GET /local/metrics` |
| `sessions(model_id, limit, db)` | Route | `GET /local/sessions` |
| `sessions_texts(model_id, limit, db)` | Route | `GET /local/sessions/texts` |
| `ngram_export(model_id)` | Route | `GET /local/ngram/export` |
| `ngram_save(model_id)` | Route | `POST /local/ngram/save` |

---

### `Server/dl_module/hybrid_model.py`

**Classes:**

| Class | Role |
|---|---|
| `_WordLSTM` | `nn.Module`: 2-layer embedding + LSTM + linear head. `embed_dim=96`, `hidden_dim=160` |
| `LstmPrediction` | Dataclass: `word: str`, `score: float` |
| `LstmWordModel` | Wrapper around `_WordLSTM`. Manages vocab, inference, training, weight serialisation |
| `HybridModelAgent` | Combines `LstmWordModel` + 2 × `NgramModel` (local + global) with confidence-gated blend |

**Key functions:**

| Function | Class | Description |
|---|---|---|
| `_build_vocab(max_vocab=2500)` | module | Builds word↔index from seed corpus at import time |
| `_clean_word(word)` | module | Extracts first alphanumeric token, lowercased |
| `LstmWordModel.__init__()` | | Calls `_build_vocab()`, `_init_torch()`, `_warmup()` |
| `LstmWordModel._init_torch()` | | Lazy torch import; builds `_WordLSTM` + `AdamW` optimizer |
| `LstmWordModel._warmup()` | | Runs dummy forward pass to pre-compile LSTM kernels |
| `LstmWordModel.predict(context_words, k)` | | Returns `(list[LstmPrediction], max_confidence)` |
| `LstmWordModel.train_texts(texts)` | | Fine-tunes on texts; CrossEntropyLoss; grad clip 1.0; returns step count |
| `LstmWordModel.state_payload()` | | Serialises weights as gzip+base64 JSON |
| `LstmWordModel.apply_peer_state(payload, mix)` | | L2-normalised weighted blend of peer weights |
| `LstmWordModel.reinitialise()` | | Forces re-import of torch and model rebuild |
| `HybridModelAgent.predict(context_text, cursor_pos, k)` | | Main inference: confidence gate → score blend → ranked list |
| `HybridModelAgent.observe(text, train_lstm)` | | Updates local n-gram; optionally fine-tunes LSTM |
| `HybridModelAgent.train_texts(texts)` | | Batch training on historic texts |
| `HybridModelAgent.update_global(peer_model)` | | Merges peer `NgramModel` into `self.global_` |
| `HybridModelAgent.apply_federated_payload(payload)` | | Merges n-gram delta + optional LSTM weights from peer payload |
| `HybridModelAgent.get_local_delta(top_k)` | | Extracts top-k n-grams as a compact sharing dict |
| `HybridModelAgent.get_federated_payload(top_k)` | | Builds full hybrid sharing payload (n-gram + LSTM state) |
| `HybridModelAgent.to_persistence_dict()` | | Serialises local + global n-gram state + versions |
| `HybridModelAgent.from_persistence_dict(data)` | | Restores state from persistence dict |

**Call chain -- prediction:**

```
POST /local/predict
  -> ModelSlot.predict()
       -> CacheAgent.lookup()
       -> HybridModelAgent.predict()
            -> tokenize(prefix)
            -> [if autocomplete] NgramModel.autocomplete() x2 (local, global)
            -> [if next_word]
                 -> LstmWordModel.predict()
                      -> _WordLSTM forward pass
                      -> torch.softmax()
                      -> torch.topk()
                 -> NgramModel.predict() (local)  [trigram->bigram->unigram backoff]
                 -> NgramModel.predict() (global)
            -> max-normalise each source
            -> confidence gate: lstm_conf >= CONF_THRESHOLD ?
            -> weighted blend (confident or fallback branch)
            -> sort by score, return top-k
       -> blend with cache scores (0.25) + recency (0.10)
       -> PersonalizationLayer.rerank()
            -> RecentWordCache.score() * ALPHA (0.15)
            -> UserDictionary.score() * BETA (0.20)
       -> filter max_chars, return top-k Suggestion objects
```

**Call chain -- session end / training:**

```
POST /local/session/end
  -> redact_text(final_text)
  -> ModelSlot.observe_text()  [PersonalizationLayer.observe()]
  -> asyncio.create_task(TrainingAgent.on_sentence_complete())
       -> HybridModelAgent.observe()   [NgramModel.observe() + LstmWordModel.train_texts()]
       -> CacheAgent.observe()
       -> FederatedSyncAgent.flush_into_global()
       -> [if auto_share] TrainingAgent._share_in_background()
            -> FederatedSyncAgent.prepare_outbound()
            -> FederatedSyncAgent.share_delta()  [POST /federated/delta to each peer]
```

---

### `Server/dl_module/trie_model.py`

| Class / Function | Role |
|---|---|
| `TrieNode` | Compact prefix-tree node: `children`, `count`, `total` |
| `NgramModel` | Trigram trie LM; Laplace smoothing (`SMOOTHING=0.5`); vocab cap 1000 |
| `NgramModel.observe(words)` | Records all unigram/bigram/trigram transitions |
| `NgramModel._maybe_evict(incoming)` | Evicts least-frequent word when vocab cap reached |
| `NgramModel.predict(context_words, k)` | Trigram → bigram → unigram backoff |
| `NgramModel.autocomplete(prefix, k)` | Prefix search over unigram vocab |
| `NgramModel.merge(other)` | Federated count-sum merge |
| `NgramModel.to_dict() / from_dict()` | JSON-serialisable persistence |
| `tokenize(text)` | Lowercase regex tokeniser: `[a-zA-Z']+|[0-9]+` |
| `build_seed_model()` | Returns `NgramModel` seeded from `Data/seed_corpus.txt` |
| `_load_seed_texts()` | Reads seed corpus; skips `#` comments and blank lines |

---

### `Server/dl_module/cache_agent.py`

| Class / Function | Role |
|---|---|
| `CacheAgent` | LRU phrase cache (capacity 512) + sliding word window (capacity 256) |
| `CacheAgent.observe(text)` | Learns bigram/trigram transitions + word frequency |
| `CacheAgent.lookup(words, k)` | Trigram then bigram context lookup; returns empty on miss |
| `CacheAgent.recency_score(word)` | Normalised recency in [0, 1] |
| `CacheAgent.stats()` | Returns `{phrase_keys, word_window_size, unique_words}` |

---

### `Server/dl_module/personalization.py`

| Class / Function | Role |
|---|---|
| `RecentWordCache` | Sliding window (capacity 128); `score()` returns normalised frequency |
| `UserDictEntry` | Dataclass: `word`, `weight`, `category`, `created_at`, `use_count` |
| `UserDictionary` | Dict of `UserDictEntry`; `add()`, `remove()`, `score()`, `to_list()` |
| `PersonalizationLayer` | `rerank(candidates, base_scores)` adds ALPHA × recency + BETA × dict weight |
| `PersonalizationLayer.save(path)` | Persists `cache_window` + `user_dict` to JSON |
| `PersonalizationLayer.load(path)` | Restores state; silent on missing file |

---

### `Server/model_registry.py`

| Class / Function | Role |
|---|---|
| `Suggestion` | Dataclass: `text`, `score`, `source`, `mode`, `partial_word`, `lstm_conf` |
| `ModelSlot.__init__(model_id, data_dir)` | Creates `HybridModelAgent`, `CacheAgent`, `PersonalizationLayer`; calls `_load()` |
| `ModelSlot.predict(context_text, cursor_pos, k, max_chars)` | Full prediction pipeline (cache + model + personalisation) |
| `ModelSlot.observe_text(text)` | Delegates to `PersonalizationLayer.observe()` |
| `ModelSlot.save()` | Writes hybrid state JSON, LSTM weights JSON, personalisation JSON |
| `ModelSlot._load()` | Restores all state files; falls back to legacy n-gram path |
| `ModelSlot.get_delta_payload(top_k)` | Returns `HybridModelAgent.get_federated_payload()` |
| `ModelSlot.apply_peer_delta(delta_data)` | Calls `HybridModelAgent.apply_federated_payload()` |
| `ModelRegistry.get(model_id)` | Lazy-creates and caches `ModelSlot` by ID |
| `ModelRegistry.save_all()` | Persists all loaded slots (best-effort) |

**Persistence file layout per model slot:**

```
<data_dir>/
    hybrid_<model_id>.json    -- n-gram local+global state + versions
    lstm_<model_id>.json      -- LSTM weights (gzip+base64)
    pers_<model_id>.json      -- personalisation recency + user dict
    device_id.txt             -- stable UUID for this node
    trainer.sqlite3           -- SQLite database (WAL mode)
```

---

### `FL/gossip.py`

| Class / Function | Role |
|---|---|
| `PeerRecord` | Dataclass: `url`, `peer_id`, `score`, `successes`, `failures`, `avg_latency_ms`, `last_seen`, `model_version` |
| `PeerReputation` | In-memory peer score tracker; EMA latency; score ∈ [0.0, 2.0] |
| `PeerReputation.record_success(url, latency_ms)` | Boosts score by `SUCCESS_BOOST` (0.1) minus latency penalty |
| `PeerReputation.record_failure(url)` | Reduces score by `FAILURE_PENALTY` (0.2) |
| `PeerReputation.sorted_peers(min_score)` | Returns peers above threshold, highest first |
| `GossipEngine.run_round(model_id, peer_urls, get_slot_fn, timeout_s)` | Tries peers in reputation order; stops on first successful merge |
| `GossipEngine._fetch_delta(peer_url, model_id, timeout_s)` | `GET /federated/weights?model_id=<id>` on target peer |

---

### `FL/federated_sync.py`

| Class / Function | Role |
|---|---|
| `FederatedSyncAgent.prepare_outbound(model_agent, model_id)` | Builds envelope: `{device_id, model_id, delta, ts}` |
| `FederatedSyncAgent.share_delta(payload, peer_urls, semaphore)` | Concurrent `POST /federated/delta` to all peers |
| `FederatedSyncAgent.receive_delta(payload)` | Enqueues inbound delta (not used by any current code path) |
| `FederatedSyncAgent.flush_into_global(model_agent)` | Drains `_pending_deltas`; calls `apply_federated_payload()` for each |
| `FederatedSyncAgent.fetch_peer_delta(peer_url, model_id)` | Single-peer pull (used internally by `GossipEngine`) |

---

### `FL/training_agent.py`

| Class / Function | Role |
|---|---|
| `TrainingAgent.on_sentence_complete(text, model_id, peer_urls, auto_share)` | Async post-session pipeline: broadcast → observe → cache → flush → optionally share |
| `TrainingAgent._share_in_background(model_id, peer_urls)` | Background task: `prepare_outbound` → `share_delta`; never blocks predictions |
| `TrainingAgent.manual_train(texts)` | Synchronous batch training; calls `train_texts()` then `cache.observe()` |

---

### `Server/peer_discovery.py`

| Class / Function | Role |
|---|---|
| `DiscoveredPeer` | Dataclass: `url`, `peer_id`, `latency_ms`, `status`, `meta`, `last_seen` |
| `PeerDiscovery.scan(enable_lan)` | Concurrent `GET /health` probes on all candidate URLs |
| `PeerDiscovery.get_active_urls()` | Returns URLs of peers with `status == "ok"` |
| `PeerDiscovery._build_candidate_urls(enable_lan)` | Generates localhost:8001-8020 + optional /24 LAN |
| `_probe(url, timeout_s)` | Single-URL probe; returns `DiscoveredPeer` or `None` |
| `_local_ipv4_addresses()` | Detects non-loopback local IPv4 addresses for LAN scanning |
| `_validate_port_range(ports)` | Enforces 8001-8020 range |

---

### `Server/db.py`

| Class / Function | Role |
|---|---|
| `SqliteDB.__init__(path)` | Opens connection; sets WAL + NORMAL sync |
| `SqliteDB.execute(sql, params)` | Write + immediate commit, locked |
| `SqliteDB.query_one(sql, params)` | Returns first row as dict or None |
| `SqliteDB.query_all(sql, params)` | Returns all rows as list of dicts |
| `_create_core_tables(db)` | Creates: `config`, `models_registry`, `sessions`, `events`, `weights` |
| `init_db(db)` | Calls `_create_core_tables()` then `run_migrations()` |
| `now_ts()` | `int(time.time())` |
| `json_dumps(value)` | Canonical compact JSON (sort_keys, no spaces) |
| `json_loads(value)` | `json.loads()` |

---

### `Server/db_migrations.py`

| Migration | Version | Change |
|---|---|---|
| `_migration_1` | 1 | Drops legacy `profiles` table; removes preset model rows |
| `_migration_2` | 2 | Creates `peer_reputation` table |
| `_migration_3` | 3 | Creates `personalization_state` table |
| `_migration_4` | 4 | Creates `ngram_sync_log` table |

---

## Data Flow Section

### Inbound federated delta (gossip pull path)

```
GossipEngine.run_round()
  -> GossipEngine._fetch_delta()           GET /federated/weights (peer)
  -> ModelSlot.apply_peer_delta()
       -> HybridModelAgent.apply_federated_payload(payload)
            -> NgramModel.from_dict()       build peer model from delta
            -> NgramModel.merge()           count-sum into global_
            -> LstmWordModel.apply_peer_state(mix=0.35)
                 -> gzip decompress + torch.load
                 -> L2-normalise each peer tensor
                 -> (1-mix)*own + mix*peer_normalised
```

### Outbound federated delta (push path)

```
FederatedSyncAgent.prepare_outbound(model_agent, model_id)
  -> HybridModelAgent.get_federated_payload(top_k=200)
       -> get_local_delta()    prune local n-gram to top-k per context
       -> LstmWordModel.state_payload()   gzip+base64 weights
  -> returns {device_id, model_id, delta: {version, ngram, lstm_state}, ts}

FederatedSyncAgent.share_delta(payload, peer_urls, semaphore)
  -> concurrent POST /federated/delta to each peer
```

### Config read/write flow

```
_cfg_get(key, default)
  -> SELECT value_json FROM config WHERE key=? -> json_loads()

_cfg_set(key, value)
  -> json_dumps(value) -> INSERT ... ON CONFLICT DO UPDATE SET ...

Settings are read fresh on every relevant request (no in-process cache).
Class-level attributes on LstmWordModel and HybridModelAgent are mutated
directly by update_settings() to apply changes immediately without restart.
```
