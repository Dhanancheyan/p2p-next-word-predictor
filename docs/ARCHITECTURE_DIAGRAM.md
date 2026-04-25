# ARCHITECTURE_DIAGRAM.md -- NWP v12 ASCII Diagrams

## 1. System Architecture

```
+---------------------------------------------------------------+
|                        NWP v12 Node                          |
|                                                               |
|  +----------+    HTTP/WebSocket    +----------------------+   |
|  | Frontend | <-----------------> |   FastAPI (app.py)   |   |
|  | (Browser)|                     |                      |   |
|  +----------+                     |  Routes:             |   |
|                                   |  /local/*            |   |
|                                   |  /federated/*        |   |
|                                   |  /gossip/*           |   |
|                                   |  /health             |   |
|                                   |  /ws/local (WS)      |   |
|                                   +----------+-----------+   |
|                                              |               |
|              +-------------------------------+               |
|              |                                               |
|  +-----------v-----------+   +---------------------------+   |
|  |    ModelRegistry       |   |       SqliteDB            |   |
|  |  (lazy-loading slots)  |   |  config / sessions /      |   |
|  +-----------+-----------+   |  events / models_registry  |   |
|              |               +---------------------------+   |
|  +-----------v-----------+                                   |
|  |      ModelSlot         |                                   |
|  |  +------------------+ |                                   |
|  |  | HybridModelAgent | |                                   |
|  |  |  - LstmWordModel | |                                   |
|  |  |  - NgramModel(L) | |                                   |
|  |  |  - NgramModel(G) | |                                   |
|  |  +------------------+ |                                   |
|  |  +------------------+ |                                   |
|  |  |   CacheAgent     | |                                   |
|  |  +------------------+ |                                   |
|  |  +------------------+ |                                   |
|  |  |PersonalizationLyr| |                                   |
|  |  +------------------+ |                                   |
|  +-----------+-----------+                                   |
|              |                                               |
|  +-----------v-----------+                                   |
|  |   Disk (data_dir)     |                                   |
|  |  hybrid_<id>.json     |                                   |
|  |  lstm_<id>.json       |                                   |
|  |  pers_<id>.json       |                                   |
|  |  device_id.txt        |                                   |
|  |  trainer.sqlite3      |                                   |
|  +-----------------------+                                   |
|                                                               |
|  Background asyncio tasks:                                    |
|  _auto_discovery_loop (every 60 s)                           |
|  _gossip_loop         (every gossip_interval_s, default 300) |
|  _flush_loop          (every 30 s)                           |
+---------------------------------------------------------------+

           HTTP (8001-8020)                HTTP (8001-8020)
NWP Node A <------------------------------> NWP Node B
           GET /federated/weights
           POST /federated/delta
           GET /health
```

---

## 2. Prediction Data Flow Pipeline

```
POST /local/predict
{model_id, context_text, cursor_pos, k, max_chars}
        |
        v
  ModelSlot.predict()
        |
        +---> [1] CacheAgent.lookup(context_words, k)
        |           LRU phrase cache; trigram -> bigram lookup
        |           returns: [(word, score), ...]  or  []
        |
        +---> [2] HybridModelAgent.predict(context_text, cursor_pos, k*3)
        |           |
        |           +-- tokenize prefix; detect partial_word
        |           |
        |           +-- if autocomplete (cursor mid-word):
        |           |     NgramModel(local).autocomplete(prefix, k*4)
        |           |     NgramModel(global).autocomplete(prefix, k*4)
        |           |     --> no LSTM in autocomplete mode
        |           |
        |           +-- if next_word:
        |           |     LstmWordModel.predict(context_words, k*4)
        |           |       -> _WordLSTM forward pass
        |           |       -> softmax -> topk
        |           |       -> returns (predictions, max_confidence)
        |           |     NgramModel(local).predict(context_words, k*4)
        |           |       -> trigram -> bigram -> unigram backoff
        |           |     NgramModel(global).predict(context_words, k*4)
        |           |
        |           +-- max-normalise each source dict
        |           |
        |           +-- confidence gate:
        |           |     lstm_conf >= CONF_THRESHOLD (0.05)?
        |           |
        |           +-- [YES - confident branch]:
        |           |     score = 0.50*lstm + 0.20*local + 0.30*global
        |           |
        |           +-- [NO - fallback branch]:
        |           |     score = 0.40*local + 0.60*global
        |           |
        |           +-- sort by score, return top-k suggestions
        |
        +---> [3] Blend:
        |           model_score + 0.25 * cache_score
        |                       + 0.10 * recency_score
        |
        +---> [4] PersonalizationLayer.rerank()
        |           final = blend_score
        |                 + 0.15 * RecentWordCache.score(word)
        |                 + 0.20 * UserDictionary.score(word)
        |
        +---> filter by max_chars, slice to k
        |
        v
  JSON response: {suggestions[], latency_ms, engine, model_versions}
```

---

## 3. Distributed / Federated Flow

```
                    +-----------+
                    | NWP Node A|
                    |           |
   Session ends --> | Training  |
                    | Agent     |
                    |  observe()|
                    |  cache()  |
                    |  flush()  |
                    +-----+-----+
                          |
             auto_share? (bool, DB-backed)
                          |
          +---------------+
          |  YES                  NO
          v
  FederatedSyncAgent
  .prepare_outbound()
  .share_delta()
    |
    | POST /federated/delta
    +----------------------------> NWP Node B
    |                              .apply_peer_delta()
    |                               -> apply_federated_payload()
    +----------------------------> NWP Node C
    |                              .apply_peer_delta()
    v
  Hub.broadcast_json(share_complete)


  Background Gossip Loop (every gossip_interval_s):
  +------------------------------------------+
  |  _gossip_loop (Node A)                   |
  |                                          |
  |  GossipEngine.run_round()                |
  |    sort peers by PeerReputation.score()  |
  |    for each peer (highest score first):  |
  |      GET /federated/weights (Node B)     |
  |        <-- {ngram delta + lstm_state}    |
  |      ModelSlot.apply_peer_delta()        |
  |        -> NgramModel.merge() (global_)   |
  |        -> LstmWordModel.apply_peer_state |
  |           (L2-normalised, mix=0.35)      |
  |      record_success(peer, latency_ms)    |
  |      break (stop on first success)       |
  |    on failure: record_failure(peer)      |
  +------------------------------------------+


  Auto-Discovery Loop (every 60 s):
  +------------------------------------------+
  |  _auto_discovery_loop (Node A)           |
  |                                          |
  |  PeerDiscovery.scan()                    |
  |    probe localhost:8001-8020 concurrently|
  |    [optional] probe LAN /24:8001-8020    |
  |    match GET /health -> {ok:true}        |
  |    cache discovered peers                |
  |    update PeerReputation                 |
  |    broadcast peers_discovered (WS)       |
  +------------------------------------------+


  Flush Loop (every 30 s):
  +------------------------------------------+
  |  _flush_loop                             |
  |                                          |
  |  for each enabled model:                 |
  |    FederatedSyncAgent.flush_into_global()|
  |      drain _pending_deltas queue         |
  |      apply_federated_payload() each      |
  |    if merged > 0:                        |
  |      broadcast global_model_updated (WS) |
  +------------------------------------------+
```

---

## 4. LSTM Weight Merge (L2-Normalised Federated Blend)

```
  Local model tensor (own):         peer tensor (incoming):
  shape [D]                         shape [D]  (must match)
      |                                 |
      |                        peer_norm = ||peer||_2
      |                        peer_normalised = peer / (peer_norm + 1e-8)
      |                                          * ||own||_2
      |                                 |
      +---------(1 - mix) * own --------+
      |         + mix * peer_normalised |
      |                                 |
      v
  merged tensor (loaded back into model)

  mix = 0.35 (gossip rounds)
  mix = 1.0  (direct upload, local save/load)
```

---

## 5. Post-Session Training Pipeline

```
POST /local/session/end
{session_id, final_text}
        |
        v
  redact_text(final_text)
  [email -> <EMAIL>, URL -> <URL>, IP -> <IP>, 3+digits -> <NUM>]
        |
        v
  sessions table updated (end_ts, text_redacted)
        |
        v
  ModelSlot.observe_text(redacted)   [PersonalizationLayer only]
        |
        v
  asyncio.create_task(TrainingAgent.on_sentence_complete)
        |
        +-- broadcast training_started (WS)
        |
        +-- HybridModelAgent.observe(text)
        |     NgramModel(local).observe(words)   [sync, <1 ms]
        |     LstmWordModel.train_texts([text])  [up to TRAIN_STEPS SGD steps]
        |
        +-- CacheAgent.observe(text)
        |
        +-- FederatedSyncAgent.flush_into_global()
        |     drain _pending_deltas
        |
        +-- broadcast training_complete (WS)
        |
        +-- [if auto_share and peers exist]
              asyncio.create_task(_share_in_background)
                -> POST /federated/delta to each peer
                -> broadcast share_complete (WS)
```
