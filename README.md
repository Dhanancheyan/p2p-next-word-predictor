# P2P Next-Word Predictor

A privacy-preserving, federated next-word prediction system that runs entirely on your local machine — no cloud, no data leaving your device.

Built as a hybrid **LSTM + n-gram** language model with a **peer-to-peer gossip federated learning** layer, it lets multiple instances share learned patterns without ever sharing raw text.

---

## What it does

Type in the browser UI and the server predicts your next word in real time. Over time, the model learns your writing patterns locally. Run multiple instances on the same machine or across a LAN, and they collaboratively improve each other through federated learning — sharing only compressed model deltas, never your actual typed text.

**Key properties:**

- **Hybrid prediction** — LSTM neural network + local n-gram trie + peer-contributed global n-gram, confidence-gated and blended at inference time
- **On-device training** — session text trains the model locally after each session ends; no epoch wait, sub-second trie updates
- **P2P gossip federated learning** — no central server; peers discover and sync with each other automatically
- **Personalisation layer** — recency cache and user dictionary re-rank suggestions to your vocabulary without retraining
- **Privacy by design** — all text is redacted (emails, URLs, IPs, digit runs) before storage; raw text never leaves the device
- **PyTorch-optional** — falls back to n-gram-only mode if PyTorch is not installed, with zero degradation to the FL pipeline

---

## Project structure

```
p2p-next-word-predictor/
├── nwp_v12/                App package
│   ├── __main__.py         Entry point (CLI flags: --host, --port, --data-dir)
│   ├── Data/               Seed corpus and LSTM weights placeholder
│   ├── FL/                 Federated learning — gossip engine, sync agent, training agent
│   ├── Server/             FastAPI app, WebSocket hub, DB, peer discovery, model registry
│   │   └── dl_module/      Prediction engines: LSTM, n-gram trie, cache, personalisation
│   └── Frontend/           Browser UI (HTML + JS + CSS, no build step)
├── docs/                   Documentation
├── notebook/               Training notebook
├── pyproject.toml
└── .gitignore
```

---

## Quick start

**Requirements:** Python ≥ 3.10

```bash
# Clone and install
git clone https://github.com/Dhanancheyan/p2p-next-word-predictor.git
cd p2p-next-word-predictor
pip install -r requirements.txt

# Start a single instance
python -m nwp_v12 --port 8001

# Open the UI
# → http://127.0.0.1:8001
```

Optional — enable the LSTM engine (runs in n-gram-only mode without it):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## Running multiple instances

This is the primary way to use the federated learning features. Each instance is independent — it has its own data directory, SQLite database, and model state. They discover each other automatically.

### Same machine, shared codebase, separate data folders

This is the recommended setup for local FL testing. Run multiple terminals from the **same project folder**, pointing each instance to a different `--data-dir`:

```bash
# Terminal 1 — Peer A
python -m nwp_v12 --port 8001 --data-dir data/peer1

# Terminal 2 — Peer B
python -m nwp_v12 --port 8002 --data-dir data/peer2

# Terminal 3 — Peer C (optional)
python -m nwp_v12 --port 8003 --data-dir data/peer3
```

Each instance will auto-discover the others within 60 seconds by scanning localhost ports 8001–8020. Open separate browser tabs to interact with each peer independently.

### All CLI flags

| Flag | Default | Description |
|---|---|---|
| `--host` | `127.0.0.1` | Bind host |
| `--port` | `8001` | Bind port — must be in range **8001–8020** |
| `--data-dir` | `data/trainer` | Storage root for SQLite DB and model files |
| `--static-dir` | `nwp_v12/Frontend` | Path to UI assets |

### Verifying FL is working

1. Open Peer A (`http://127.0.0.1:8001`) and type a sentence with a unique word (e.g. "zorbaxword")
2. Click **End Session** on Peer A
3. In **FL Settings**, click **Share to Peers**
4. Switch to Peer B (`http://127.0.0.1:8002`) and start typing the unique word
5. The suggestion popup will show a **★ peer** badge on words learned from Peer A

Gossip also runs automatically every 5 minutes — the version counter in the header increments each time a peer delta is applied.

---

## How the prediction pipeline works

```
User input
    │
    ▼
┌─────────────┐     ┌──────────────────┐     ┌──────────────────────┐
│    LSTM      │  +  │  Local n-gram    │  +  │  Global n-gram       │
│ (word-level) │     │  (trigram trie)  │     │  (from FL peers)     │
└─────────────┘     └──────────────────┘     └──────────────────────┘
         │                   │                          │
         └───────────────────┴──────────────────────────┘
                             │
                    Confidence-gated blend
                    (LSTM conf ≥ threshold → 50% LSTM + 20% local + 30% peer)
                    (LSTM conf < threshold → local + global n-gram only)
                             │
                    Cache + personalisation re-rank
                             │
                         Predictions
```

Session phrase cache provides sub-millisecond recall of recently typed phrases, independent of the model stack.

---

## Federated learning

No central server. Every node is both a client and a server.

- **Gossip protocol** — peers exchange hybrid payloads (top-K n-gram delta + gzip-compressed LSTM weights) at a configurable interval (default: 5 minutes)
- **Delta sharing** — only the top-200 most-changed n-gram entries are sent per round, keeping payloads under 50 KB
- **LSTM merging** — incoming peer weights are L2-normalised before blending to prevent any single peer from dominating the local model
- **Peer reputation** — each peer is scored by uptime, latency, and sync success rate; low-reputation peers are deprioritised for gossip rounds
- **Peer discovery** — scans localhost ports 8001–8020 on startup and every 60 s; optional LAN /24 subnet scan available in **FL Settings**
- **Versioned aggregation** — peers reject uploads trained on a stale global version; the error message tells the client exactly which version to pull first

---

## Personalisation

The personalisation layer adapts to your typing without any retraining:

- **Recency cache** — boosts words you recently typed in the current session
- **User dictionary** — add names, domain terms, or slang via the UI; they surface above generic suggestions
- **Re-ranking formula** — `final_score = base_score + 0.15 × recency + 0.20 × user_dict_weight`
- **Source labels** — suggestions from the cache show a `(pers)` tag; n-gram fallback shows `(local)`

Manage via **Settings → Personalisation** in the UI, or via the API:

```bash
# Add a word to your user dictionary
POST /local/personalization/word/add  {"word": "zorbaxword"}

# Feed typed text to the recency cache
POST /local/personalization/observe   {"text": "..."}
```

---

## Configuration

All settings are stored in the local SQLite database and editable through the web UI under **Settings**:

| Setting | Default | Description |
|---|---|---|
| `auto_train` | `true` | Fine-tune model after each session ends |
| `auto_share` | `false` | Auto-push delta to peers after training |
| `gossip_enabled` | `true` | Run background gossip FL rounds |
| `gossip_interval_s` | `300` | Seconds between gossip rounds (30–3600) |
| `max_concurrent_peer_sync` | `10` | Max simultaneous peer push connections |
| `discovery_enable_lan` | `false` | Also scan LAN /24 subnet (slower) |
| `lstm_conf_threshold` | `0.05` | Minimum LSTM softmax probability to activate the DL branch |
| `lstm_weight` | `0.50` | LSTM contribution in confident blend |
| `local_ngram_weight` | `0.20` | Local n-gram contribution |
| `global_ngram_weight` | `0.30` | Peer n-gram contribution |

---

## Using pre-trained LSTM weights

The server runs in n-gram-only mode by default. To enable the LSTM:

1. Open `NWP_Training_Notebook.ipynb` in Google Colab or Jupyter (in the `notebook/` folder)
2. Run all cells — exports `lstm_weights.json`
3. In the app, go to **DL Tuning** tab → drag-and-drop the file, or via API:

```bash
curl -X POST http://127.0.0.1:8001/local/model/upload_weights \
  -F "model_id=default" \
  -F "file=@lstm_weights.json"
```

The server performs a shape-compatibility check before loading. If the check fails, the existing model is preserved.

---

## Runtime data layout

All data is auto-created under `--data-dir` (default: `data/trainer/`). Nothing is written outside that folder.

| Path | Contents |
|---|---|
| `trainer.sqlite3` | Sessions, events, config, peer reputation |
| `hybrid_<model_id>.json` | Local + global n-gram state |
| `lstm_<model_id>.json` | LSTM weight snapshot (gzip + base64) |
| `pers_<model_id>.json` | Personalisation layer (recency cache + user dictionary) |
| `device_id.txt` | Stable peer identity (auto-generated UUID) |

When running multiple instances with `--data-dir`, each gets its own isolated copy of this layout.

---

## API reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness, engine info, peer ID, model version |
| `POST` | `/local/predict` | Next-word prediction |
| `POST` | `/local/session/start` | Start a typing session |
| `POST` | `/local/session/end` | End session + trigger training |
| `POST` | `/local/train` | Manual training on recent sessions |
| `POST` | `/local/share` | Push delta to all discovered peers |
| `POST` | `/local/pull_global` | Pull from highest-reputation peer |
| `GET` | `/local/settings` | Read current settings |
| `POST` | `/local/settings` | Update settings |
| `POST` | `/local/model/upload_weights` | Upload external LSTM weights |
| `POST` | `/local/personalization/word/add` | Add word to user dictionary |
| `POST` | `/local/personalization/observe` | Feed text to recency cache |
| `GET` | `/local/personalization/words` | List user dictionary |
| `POST` | `/local/peers/scan` | Trigger immediate peer scan |
| `GET` | `/local/peers/reputation` | Peer reputation scores |
| `GET` | `/federated/weights` | Serve hybrid payload to peers |
| `POST` | `/federated/delta` | Receive inbound peer delta |
| `POST` | `/gossip/round` | Trigger one manual gossip round |
| `WS` | `/ws/local` | Live events (training, sync, hot-swap) |

---

## Diagnostics

```bash
# Check PyTorch installation and device detection
python scripts/doctor.py

# End-to-end FL smoke test (starts 2 local instances, exchanges a phrase, verifies pipeline)
python scripts/smoke_fl_round.py
```

---

## Tech stack

| Layer | Technology |
|---|---|
| Server | FastAPI + Uvicorn |
| Real-time | WebSockets |
| Neural model | PyTorch (optional, CPU) |
| Storage | SQLite |
| P2P transport | HTTP (httpx) |
| Frontend | Vanilla HTML/JS/CSS |

---

## Docs

Detailed documentation is in the [`docs/`](./docs/) folder.

---

## Citation

If you use this project in your research, please cite it:

```bibtex
@software{dhanancheyan2026nwp,
  author    = {Dhanancheyan},
  title     = {P2P Next-Word Predictor},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/Dhanancheyan/p2p-next-word-predictor}
}
```

---

## License

MIT License. See [LICENSE](./LICENSE) for details.
