# IMPROVEMENTS.md -- NWP v12 Engineering Recommendations

---

## 1. Config System: DB-backed Class Mutation is Fragile

**Problem:** `update_settings()` mutates class-level attributes directly on `LstmWordModel` and `HybridModelAgent` (e.g., `LstmWordModel.CONF_THRESHOLD = val`). This approach ties runtime state to class globals, making it impossible to have per-model-slot weight configurations and causing unpredictable behaviour in multi-slot scenarios.

**Impact:** All model slots share the same inference weights regardless of which slot is being tuned. In a multi-model setup, tuning weights for one model affects all others.

**Recommendation:** Move `CONF_THRESHOLD`, `LSTM_WEIGHT`, `LOCAL_NGRAM_WEIGHT`, `GLOBAL_NGRAM_WEIGHT`, `LOCAL_NGRAM_FALLBACK_WEIGHT`, and `GLOBAL_NGRAM_FALLBACK_WEIGHT` from class-level attributes to per-instance attributes on `HybridModelAgent`. Thread those values through from the settings layer. Config reads in the inference path should use the per-instance values, not class constants.

---

## 2. Three DB Tables Created but Never Written

**Problem:** Migrations 2, 3, and 4 create `peer_reputation`, `personalization_state`, and `ngram_sync_log` tables. The `weights` table is created in `_create_core_tables`. None of these tables are written to by any production code path.

**Evidence:**
- `PeerReputation` is purely in-memory (`FL/gossip.py`). Its state is lost on every restart.
- `PersonalizationLayer` persists to JSON files, not the `personalization_state` table.
- `ngram_sync_log` has no corresponding write calls anywhere in the codebase.
- The `weights` table has no corresponding write calls anywhere in the codebase.

**Impact:** Schema bloat; misleading to developers who may expect these tables to contain live data. Peer reputation loss on restart means gossip quality degrades after every server restart.

**Recommendation:**
- Persist `PeerReputation` records to the `peer_reputation` table on shutdown and reload on startup.
- Either write `PersonalizationLayer` snapshots to `personalization_state` or remove the table.
- Implement `ngram_sync_log` writes in `federated_receive_delta` and `gossip.run_round` or drop the table.
- Implement weight file metadata tracking in `weights` table on LSTM save/load, or drop the table.

---

## 3. Two Parallel Inbound Delta Paths with No Coordination

**Problem:** `POST /federated/delta` calls `apply_peer_delta()` immediately. `_flush_loop()` also calls `flush_into_global()` from `FederatedSyncAgent._pending_deltas`. The `receive_delta()` method that feeds the queue is never called by any code path in the server.

**Evidence:** `FederatedSyncAgent.receive_delta()` exists but has no callers. `flush_into_global()` always processes an empty list in normal operation. The comment in `federated_sync.py` acknowledges this: "the primary inbound path in app.py calls apply_peer_delta immediately on receipt for demo visibility."

**Impact:** `_flush_loop` runs every 30 seconds but does no work. The queue-based path is dead code in practice. Developers maintaining this code may be misled about the actual merge path.

**Recommendation:** Consolidate to a single inbound path. If immediate merge is preferred, remove `_pending_deltas` queue and `_flush_loop`. If batching is desired, route `POST /federated/delta` through `receive_delta()` and let `_flush_loop` handle all merges.

---

## 4. LAN Scan Has No Rate Limiting

**Problem:** `PeerDiscovery.scan(enable_lan=True)` fires `asyncio.gather()` with up to 254 hosts × 20 ports = 5,080 concurrent HTTP requests simultaneously. There is no chunking, semaphore, or timeout beyond the per-probe `probe_timeout_s = 1.5 s`.

**Impact:** On a busy LAN, this saturates the event loop and can starve other request handling for 1.5+ seconds. It can also trigger firewall alerts or rate-limiting on the network.

**Recommendation:** Process LAN candidates in batches (e.g., 50-100 concurrent probes) using `asyncio.Semaphore`. Consider separating LAN scanning into its own background task with longer intervals.

---

## 5. `ModelRegistry` Has No Thread-Safety Guarantee for Slot Creation

**Problem:** `ModelRegistry.get()` checks `if model_id not in self._slots` and then creates a new `ModelSlot` without locking. In a concurrent FastAPI environment (multiple simultaneous requests for a new `model_id`), two threads could both pass the check and create duplicate `ModelSlot` instances, with the second overwriting the first.

**Impact:** Race on first access to any new model ID. The window is small but exists under load.

**Recommendation:** Protect slot creation with `threading.Lock()` or `asyncio.Lock()`.

---

## 6. LSTM Architecture Mismatch: Docstring Claims 2-Layer, Code Has 1-Layer

**Problem:** The module docstring and class docstring for `_WordLSTM` reference "Two-layer embedding + LSTM + linear head." The actual `nn.LSTM` instantiation does not pass `num_layers`, defaulting to 1 layer.

**Evidence:** `self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)` — no `num_layers` argument.

**Impact:** Misleading documentation. If a second layer is intended for capacity, it is missing.

**Recommendation:** Either add `num_layers=2` to match the documented architecture, or update the docstring to say single-layer LSTM.

---

## 7. `auto_train` Config Key Has No Code Effect

**Problem:** `auto_train` is stored and returned by the settings API, but no code path in the server reads it to conditionally gate training. Training always runs on session end regardless of this setting.

**Evidence:** `_cfg_get("auto_train", True)` is only called in `get_settings()` and `update_settings()`. It is never read in `session_end()` or `TrainingAgent`.

**Impact:** The UI setting has no effect on behaviour. This is a silent mis-feature.

**Recommendation:** Read `auto_train` in `session_end()` and skip `TrainingAgent.on_sentence_complete()` when it is false, or remove the setting from the API.

---

## 8. `source_model_id` in `CreateModelRequest` is Accepted but Ignored

**Problem:** `CreateModelRequest` includes a `source_model_id` field to support `clone` creation type, but `models_create()` only inserts a new registry row and calls `registry.get(new_id)` on a fresh blank slot. No cloning of source model state occurs.

**Impact:** The `clone` creation type silently behaves identically to `blank`. Frontend or API users who rely on cloning will get an empty model.

**Recommendation:** Implement actual clone logic: load the source slot, copy its JSON state files to the new model ID, then load the new slot. Alternatively, remove `creation_type: clone` from the enum and `source_model_id` from the schema.

---

## 9. Settings Normalisation Only Triggers on Partial Update

**Problem:** In `update_settings()`, the normalisation of `lstm_weight`, `local_ngram_weight`, and `global_ngram_weight` only runs `if any(x is not None for x in [req.lstm_weight, req.local_ngram_weight, req.global_ngram_weight])`. If a user updates only `lstm_weight`, the current DB values of `local_ngram_weight` and `global_ngram_weight` are read as defaults from the class attribute, not from the DB. This means a partial update can un-normalise previously manually calibrated values.

**Recommendation:** When any weight in a group is updated, read the other weights from the DB (not the class attribute) as the starting values before normalisation.

---

## 10. Version String Mismatch

**Problem:** `app.py` hardcodes `"version": "11.0"` in the `/health` response. `pyproject.toml` declares `version = "12.0.0"`. The module docstring in `__main__.py` references "v11 hybrid LSTM".

**Impact:** Peer discovery and logging produce inconsistent version signals. Makes debugging peer compatibility harder.

**Recommendation:** Read the version from `importlib.metadata.version("nwp_v12")` or a shared constant, and propagate it to all version strings.

---

## 11. `accuracy_over_time` in Metrics is Hardcoded Empty

**Problem:** `GET /local/metrics` always returns `"accuracy_over_time": []`. There is no tracking of per-session or per-epoch accuracy metrics.

**Impact:** The field is present in the API contract but carries no information.

**Recommendation:** Either compute a time-series of `accept_rate` from the `events` table grouped by day/session, or remove the field from the response to avoid misleading consumers.

---

## 12. Prediction Cache Scoring Uses Additive Blend Without Normalisation

**Problem:** `ModelSlot.predict()` adds cache and recency scores additively (`score += cache_map.get(word, 0.0) * 0.25 + self.cache.recency_score(word) * 0.10`) to the model's already-normalised scores. The model scores from `HybridModelAgent` are in the confident branch sum to at most 1.0 per word, but the additive bonuses can push total scores above 1.0 in an uncontrolled way, effectively scaling up cache-familiar words disproportionately.

**Recommendation:** Normalise the final blended scores before personalisation re-ranking to maintain a consistent score magnitude.
