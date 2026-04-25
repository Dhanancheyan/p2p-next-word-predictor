"""
Server package -- local FastAPI server for NWP v11.

Modules
-------
app              : FastAPI application factory and all HTTP/WebSocket routes.
settings         : TrainerSettings dataclass and device-ID persistence.
db               : SQLite WAL wrapper (SqliteDB) and schema initialisation.
db_migrations    : Version-guarded schema migrations.
model_registry   : ModelSlot and ModelRegistry -- model lifecycle management.
peer_discovery   : Localhost and LAN port scanning for peer detection.
ws               : WebSocket broadcast hub.

Sub-package
-----------
dl_module : LSTM and n-gram prediction engines (hybrid_model, trie_model,
            cache_agent, personalization, redact, hashing).
"""
