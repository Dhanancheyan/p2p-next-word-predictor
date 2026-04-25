# AUDIT_REPORT.md -- NWP v12 Codebase Audit

Audit scope: all files in the `nwp_v12` package as documented in CODE_MAP.md, app_config.md, IMPROVEMENTS.md, ARCHITECTURE_DIAGRAM.md, and README.md. Issues are grouped by category. Each entry states the file, the problem, its runtime impact, and the code evidence.

---

## Category 1: Dead Code and Unreachable Paths

---

### AUDIT-01: `FederatedSyncAgent.receive_delta()` has no callers

**File:** `FL/federated_sync.py`

**Problem:** `FederatedSyncAgent.receive_delta(payload)` enqueues inbound deltas into `_pending_deltas`. No code path in the server calls this method. The inbound route `POST /federated/delta` calls `apply_peer_delta()` directly on the `ModelSlot`, bypassing the queue entirely.

**Impact:** The queue-based inbound pipeline is permanently idle. `_flush_loop` drains it every 30 seconds but there is never anything to drain. Any developer who adds a call to `receive_delta()` expecting it to feed the merge pipeline will find it works, but that is not the current design, and the current design does not document which path is authoritative.

**Evidence:** `FederatedSyncAgent.receive_delta()` is listed in CODE_MAP.md with the annotation "not used by any current code path". The `_flush_loop` in `app.py` calls `flush_into_global()` which drains `_pending_deltas`. `federated_receive_delta` in `app.py` calls `registry.get(model_id).apply_peer_delta(delta)` directly.

---

### AUDIT-02: `_flush_loop` performs no useful work

**File:** `Server/app.py`

**Problem:** `_flush_loop` runs every 30 seconds for every enabled model. It calls `FederatedSyncAgent.flush_into_global()`, which drains `_pending_deltas`. Because `receive_delta()` is never called (see AUDIT-01), `_pending_deltas` is always empty. The loop executes, acquires resources, iterates over models, and does nothing.

**Impact:** Wasted asyncio scheduling overhead every 30 seconds for the lifetime of the process. More critically, the loop gives a false impression that batched delta processing is active.

**Evidence:** Direct consequence of AUDIT-01. Both issues are confirmed in the comment in `federated_sync.py`: "the primary inbound path in app.py calls apply_peer_delta immediately on receipt for demo visibility."

---

### AUDIT-03: `source_model_id` field accepted in `CreateModelRequest` but never read

**File:** `Server/app.py` (route `POST /local/models/create`)

**Problem:** `CreateModelRequest` includes `source_model_id` (string or null) to support the `clone` creation type. The `models_create()` handler inserts a new registry row and calls `registry.get(new_id)` on a fresh blank slot. The `source_model_id` field is never read in the handler body.

**Impact:** The `clone` creation type is silently identical to `blank`. A caller submitting `{"creation_type": "clone", "source_model_id": "abc123"}` receives an empty model with no error. The creation_type value `"clone"` is stored in the DB but has no operational meaning.

**Evidence:** `models_create()` is listed in CODE_MAP.md. IMPROVEMENTS.md item 8 states: "No cloning of source model state occurs."

---

### AUDIT-04: `auto_train` config key is stored but never read in training code

**File:** `Server/app.py`, `FL/training_agent.py`

**Problem:** `auto_train` is written to and read from the DB config table by `get_settings()` and `update_settings()`. It is never read in `session_end()` or `TrainingAgent.on_sentence_complete()`. Training always runs unconditionally on session end.

**Impact:** The setting is a no-op. Disabling `auto_train` via `POST /local/settings` has no effect on system behaviour. This is a silent mis-feature.

**Evidence:** `_cfg_get("auto_train", True)` is only called in the settings read/write path. The `session_end` route and `TrainingAgent.on_sentence_complete` call chain in CODE_MAP.md show no branch on this flag.

---

### AUDIT-05: `accuracy_over_time` field in metrics response is permanently empty

**File:** `Server/app.py` (route `GET /local/metrics`)

**Problem:** The `metrics()` handler always returns `"accuracy_over_time": []`. No code path writes to or computes this field.

**Impact:** The field is part of the API contract (documented in API_REFERENCE.md) but carries no data. Consumers who build dashboards or monitoring on this field receive a permanently empty time series with no error signal.

**Evidence:** API_REFERENCE.md for `GET /local/metrics` includes the note: "`accuracy_over_time` is always an empty list (not implemented)."

---

### AUDIT-06: `hashing.py` utilities have no verified call sites

**File:** `Server/dl_module/hashing.py`

**Problem:** `sha256_hex()`, `canonical_json_dumps()`, and `sha256_hex_of_canonical_json()` are defined in the module. No call sites for these functions appear in CODE_MAP.md's call chains or data flow sections.

**Impact:** If these utilities are unused, the module is dead code. If they are used in paths not captured in the documentation (e.g., inside federated payload validation), the audit cannot confirm integrity-check coverage. Either way, the functions are undocumented relative to their callers.

**Evidence:** `hashing.py` is listed in CODE_MAP.md File Index with role "SHA-256 utilities" but no function in any other module is shown calling into it.

---

## Category 2: DB Tables Created but Never Written

---

### AUDIT-07: `weights` table never written

**File:** `Server/db.py` (`_create_core_tables`)

**Problem:** The `weights` table is created in the core schema with columns for `model_id`, `kind`, `version`, and an index `idx_weights_model_kind_version`. No production code path performs `INSERT` or `UPDATE` on this table.

**Impact:** The table exists in every deployed database but contains no data. Developers inspecting the schema may assume weight file metadata is tracked here. LSTM weights are saved as flat JSON files (`lstm_<id>.json`), not recorded in the DB.

**Evidence:** app_config.md SQLite table list: "weights -- Metadata for weight files (created in schema, never written by server code)."

---

### AUDIT-08: `peer_reputation` table never written; in-memory state lost on restart

**File:** `Server/db_migrations.py` (migration 2), `FL/gossip.py`

**Problem:** Migration 2 creates the `peer_reputation` table. `PeerReputation` (in `FL/gossip.py`) stores all peer score state in-memory using a plain dict. No code persists this state to the table, and no code loads from it on startup.

**Impact:** All reputation scores, success/failure counts, and latency averages are lost on every server restart. After a restart, the gossip engine treats all peers as equal (default score 1.0), degrading peer selection quality until enough rounds are completed to rebuild the scores. For a system expected to run continuously, restart-induced score loss is a meaningful quality regression.

**Evidence:** CODE_MAP.md: "`PeerReputation` -- In-memory peer score tracker." app_config.md: "peer_reputation -- Migration 2; never written by server code (state is in-memory only)." README.md Notes section confirms: "The `peer_reputation` table created by migration 2 is never written to by the server."

---

### AUDIT-09: `personalization_state` table never written; JSON files used instead

**File:** `Server/db_migrations.py` (migration 3), `Server/dl_module/personalization.py`

**Problem:** Migration 3 creates the `personalization_state` table. `PersonalizationLayer` saves and loads its state from `pers_<model_id>.json` files. The DB table is never written to.

**Impact:** Two competing persistence strategies exist for personalisation data: the DB table (empty, unused) and JSON files (active). A developer tasked with adding cross-model personalisation export would likely query the DB table and find nothing.

**Evidence:** CODE_MAP.md: "`PersonalizationLayer.save(path)` -- Persists `cache_window` + `user_dict` to JSON." app_config.md: "personalization_state -- Migration 3; never written by server code (state is in JSON files)."

---

### AUDIT-10: `ngram_sync_log` table never written

**File:** `Server/db_migrations.py` (migration 4)

**Problem:** Migration 4 creates the `ngram_sync_log` table. No code path performs any write to this table.

**Impact:** Intended to log federated n-gram sync events for auditing or debugging, it provides neither. The table occupies schema space and creates developer confusion about whether sync events are tracked.

**Evidence:** app_config.md: "ngram_sync_log -- Migration 4; never written by server code."

---

## Category 3: Inconsistencies and Mismatches

---

### AUDIT-11: Version string mismatch across three locations

**File:** `Server/app.py`, `__main__.py`, `pyproject.toml`

**Problem:** Three version declarations exist with three different values:
- `app.py` `GET /health` response: `"version": "11.0"`
- `__main__.py` module docstring: references "v11 hybrid LSTM"
- `pyproject.toml`: `version = "12.0.0"`

**Impact:** Peer discovery uses the `/health` response version for compatibility signalling. Peers running v12 will advertise themselves as v11, creating ambiguity in multi-node deployments. Log analysis and debugging are made harder when version signals are inconsistent.

**Evidence:** README.md Notes section: "The version string in `app.py` reads `"version": "11.0"` while `pyproject.toml` declares `version = "12.0.0"`. The module docstring in `__main__.py` references `v11 hybrid LSTM`."

---

### AUDIT-12: LSTM architecture mismatch between docstring and code

**File:** `Server/dl_module/hybrid_model.py`

**Problem:** The `_WordLSTM` class docstring states "Two-layer embedding + LSTM + linear head." The actual `nn.LSTM` instantiation does not pass `num_layers`, which defaults to 1.

**Impact:** Documentation-code divergence. If the second layer was intentional (for increased capacity), it is absent. If it was never intended, peer weight compatibility documentation is misleading for anyone attempting to write a compatible external model.

**Evidence:** app_config.md Model Parameters table: "Layers -- Embedding → LSTM (2-layer not explicit, 1 in code)." IMPROVEMENTS.md item 6.

---

### AUDIT-13: `weight_version` not incremented on gossip-merged LSTM weights

**File:** `Server/model_registry.py`, `Server/dl_module/hybrid_model.py`

**Problem:** `ModelSlot.weight_version` is a property that delegates to `HybridModelAgent.weight_version`. This counter is incremented in `apply_federated_payload()` when an LSTM merge succeeds, and in the direct upload path. It is not incremented in `_flush_loop`. Because the flush loop's inbound path is already dead (AUDIT-02), the practical effect is that the counter only reflects gossip-initiated merges, not any other merge path.

**Impact:** `model_versions.weights` in prediction responses does not accurately reflect all LSTM state changes. External tooling or UI components that use `weight_version` to detect model staleness may miss updates.

**Evidence:** README.md Notes: "The `weight_version` tracker is incremented on LSTM weight uploads but the `_flush_loop` does not increment it."

---

### AUDIT-14: `POST /local/logs` is an undocumented alias with divergent naming

**File:** `Server/app.py`

**Problem:** `POST /local/logs` is implemented as a direct alias for `POST /local/session/event`. It uses a different name but is not documented as an alias in a way that clarifies it is identical. The API_REFERENCE.md entry says "Alias for `POST /local/session/event`. Used by the frontend logging path" but does not confirm that the request/response schema is identical.

**Impact:** Low direct impact. Minor maintenance risk: a developer modifying `session_event` must remember to verify the alias still applies. If the alias is ever broken or diverged, frontend logging silently fails.

**Evidence:** CODE_MAP.md: `logs(req, db)` — "alias for session/event." API_REFERENCE.md `POST /local/logs` section.

---

### AUDIT-15: `ModelRegistry.get()` has a race condition on first slot creation

**File:** `Server/model_registry.py`

**Problem:** `ModelRegistry.get(model_id)` checks `if model_id not in self._slots` and then creates a new `ModelSlot`. There is no lock around this check-create sequence. Under concurrent requests for the same new `model_id`, two threads can both pass the check and create duplicate `ModelSlot` instances; the second will overwrite the first, discarding any state the first may have partially initialised.

**Impact:** The race window is narrow and only occurs on the first request to an unloaded model ID. In practice this is unlikely under typical keyboard usage but is a real hazard under load testing or automated API access.

**Evidence:** IMPROVEMENTS.md item 5: "Race on first access to any new model ID. The window is small but exists under load."

---

## Category 4: Incomplete Pipelines

---

### AUDIT-16: PII redaction applied only at session end, not to mid-session event payloads

**File:** `Server/app.py` (routes `POST /local/session/event`, `POST /local/logs`)

**Problem:** `redact_text()` is called on `final_text` in `POST /local/session/end`. Event payloads logged mid-session via `POST /local/session/event` are stored to the `events` table without PII redaction. The `payload` field of an event can contain arbitrary user-supplied JSON, including typed text fragments.

**Impact:** PII (emails, URLs, IPs, numeric sequences) present in mid-session event payloads is persisted to the `events` table unredacted, contradicting the privacy guarantee stated in the project overview ("PII redaction before any local storage").

**Evidence:** README.md Key Features: "PII redaction (email, URL, IP, numeric sequences) before any local storage or training." README.md Notes: "Session texts are stored as `text_redacted` but the redaction is applied only at session end, not to events logged mid-session."

---

### AUDIT-17: `clone` creation type produces a blank model silently

**File:** `Server/app.py` (route `POST /local/models/create`)

**Problem:** This is the functional consequence of AUDIT-03. When `creation_type = "clone"` is submitted, no error is returned and no clone operation occurs. The caller receives a success response with a new `model_id` for what is actually a blank model.

**Impact:** Any integration or UI that depends on model cloning (e.g., for A/B personalisation or backup) receives an empty model without warning. The feature is documented in the API but non-functional.

**Evidence:** API_REFERENCE.md `POST /local/models/create`: "Note: `source_model_id` is accepted in the request schema but is not used in the model creation logic."

---

### AUDIT-18: Settings normalisation reads class defaults, not DB, when partially updating weights

**File:** `Server/app.py` (`update_settings`)

**Problem:** When only one weight in a normalisation group is supplied (e.g., only `lstm_weight`), the handler reads the other weights from the class-level default attributes rather than from the DB. If the DB values were previously manually calibrated to non-default values, a partial update resets the uncalibrated weights back to class defaults before normalisation.

**Impact:** Partially updating weights silently discards prior calibration. A user who sets `local_ngram_weight=0.35, global_ngram_weight=0.15` via one request and then updates only `lstm_weight` via a second request will have the first two values overwritten.

**Evidence:** IMPROVEMENTS.md item 9: "A partial update can un-normalise previously manually calibrated values."

---

### AUDIT-19: LAN peer scan has no concurrency limit

**File:** `Server/peer_discovery.py` (`PeerDiscovery.scan`)

**Problem:** When `enable_lan=True`, `scan()` issues `asyncio.gather()` over up to 254 × 20 = 5,080 concurrent HTTP probes simultaneously. There is no `asyncio.Semaphore` or batching.

**Impact:** During a LAN scan, the event loop is saturated for up to `probe_timeout_s = 1.5` seconds, starving concurrent prediction requests. On the network, 5,080 simultaneous SYN packets may trigger firewall rate limiting or intrusion detection alerts.

**Evidence:** IMPROVEMENTS.md item 4: "fires `asyncio.gather()` with up to 254 hosts × 20 ports = 5,080 concurrent HTTP requests simultaneously."

---

## Category 5: Config and State Design Issues

---

### AUDIT-20: Settings mutate class-level attributes, making per-slot configuration impossible

**File:** `Server/app.py` (`update_settings`), `Server/dl_module/hybrid_model.py`

**Problem:** `update_settings()` mutates `LstmWordModel.CONF_THRESHOLD`, `LstmWordModel.TRAIN_STEPS`, and the weight constants directly on the class (not on any instance). All `ModelSlot` instances share these class-level values. Changing inference weights for one model changes them for all models.

**Impact:** In a multi-slot deployment (multiple named models), tuning the confidence threshold or blend weights for one model unintentionally modifies every other loaded model. There is no per-slot weight configuration.

**Evidence:** IMPROVEMENTS.md item 1: "All model slots share the same inference weights regardless of which slot is being tuned."

---

### AUDIT-21: `device_id` stored as plain text file, inconsistent with DB-backed config pattern

**File:** `Server/settings.py`

**Problem:** All runtime configuration is stored in the SQLite `config` table. The `device_id` is stored separately as `device_id.txt` in the data directory. This breaks the single-source-of-truth pattern established by the DB config system.

**Impact:** If the data directory path changes or the file is deleted, the node gets a new identity and will be treated as an unknown peer by nodes that tracked the previous ID. There is no migration or reconciliation path.

**Evidence:** app_config.md: "`device_id` -- `device_id.txt` in data_dir -- Stable UUID generated on first run, persisted as plain text." All other settings are in the `config` table.

---

### AUDIT-22: Prediction cache blend adds scores without bounding the total

**File:** `Server/model_registry.py` (`ModelSlot.predict`)

**Problem:** After `HybridModelAgent.predict()` returns scores (each in [0, 1] after max-normalisation), `ModelSlot.predict()` adds `0.25 * cache_score` and `0.10 * recency_score` directly. These additive bonuses can push individual word scores above 1.0. `PersonalizationLayer.rerank()` then adds further `ALPHA * recency` (0.15) and `BETA * user_dict` (0.20) bonuses on top. The final scores returned to callers have no upper bound and no normalisation.

**Impact:** Score magnitudes in API responses are inconsistent and grow with cache/dict familiarity, making scores incomparable across calls. The `score` field in the suggestion response is not a probability or a bounded value, yet the response schema does not document this.

**Evidence:** IMPROVEMENTS.md item 12: "The additive bonuses can push total scores above 1.0 in an uncontrolled way."

---

## Summary Table

| ID | Category | File(s) | Severity | Type |
|---|---|---|---|---|
| AUDIT-01 | Dead Code | `FL/federated_sync.py` | Medium | Unreachable method |
| AUDIT-02 | Dead Code | `Server/app.py` | Low | Idle background loop |
| AUDIT-03 | Dead Code | `Server/app.py` | Medium | Accepted-but-ignored field |
| AUDIT-04 | Dead Code | `Server/app.py`, `FL/training_agent.py` | Medium | No-op config key |
| AUDIT-05 | Dead Code | `Server/app.py` | Low | Hardcoded empty field |
| AUDIT-06 | Dead Code | `Server/dl_module/hashing.py` | Low | Unverified call sites |
| AUDIT-07 | Unused DB Table | `Server/db.py` | Low | Schema bloat |
| AUDIT-08 | Unused DB Table | `FL/gossip.py`, migrations | High | In-memory state lost on restart |
| AUDIT-09 | Unused DB Table | `personalization.py`, migrations | Low | Competing persistence strategies |
| AUDIT-10 | Unused DB Table | `db_migrations.py` | Low | Schema bloat |
| AUDIT-11 | Inconsistency | `app.py`, `__main__.py`, `pyproject.toml` | Medium | Version mismatch |
| AUDIT-12 | Inconsistency | `hybrid_model.py` | Low | Docstring/code mismatch |
| AUDIT-13 | Inconsistency | `model_registry.py`, `hybrid_model.py` | Low | Stale version counter |
| AUDIT-14 | Inconsistency | `Server/app.py` | Low | Undocumented alias |
| AUDIT-15 | Inconsistency | `Server/model_registry.py` | Medium | Race condition on slot creation |
| AUDIT-16 | Incomplete Pipeline | `Server/app.py` | High | PII in unredacted event payloads |
| AUDIT-17 | Incomplete Pipeline | `Server/app.py` | Medium | Silent no-op clone operation |
| AUDIT-18 | Incomplete Pipeline | `Server/app.py` | Medium | Partial weight update discards calibration |
| AUDIT-19 | Incomplete Pipeline | `Server/peer_discovery.py` | Medium | Unbounded LAN scan concurrency |
| AUDIT-20 | Config/State | `Server/app.py`, `hybrid_model.py` | High | Class-level mutation breaks multi-slot config |
| AUDIT-21 | Config/State | `Server/settings.py` | Low | device_id outside DB config system |
| AUDIT-22 | Config/State | `Server/model_registry.py` | Low | Unbounded score accumulation |

---

## Severity Definitions

| Severity | Meaning |
|---|---|
| High | Causes silent incorrect behaviour, data loss, or a broken privacy guarantee |
| Medium | Causes a feature to silently not work, or creates a real race/regression risk |
| Low | Documentation/schema debt; cosmetic inconsistency; no current operational impact |
