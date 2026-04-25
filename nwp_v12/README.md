# NWP v11 -- Next-Word Prediction Keyboard LM

Hybrid LSTM + n-gram federated next-word prediction server with P2P gossip FL.

---

## Directory Structure

```
nwp_v12/
  Data/           Seed corpus and LSTM weights placeholder.
  FL/             Federated learning components (gossip, sync, training pipeline).
  Server/         FastAPI server, database, model registry.
    dl_module/    Prediction engines: LSTM, n-gram trie, cache, personalisation.
  Frontend/       Web UI (HTML, JS, CSS).
  __main__.py     Server entry point.
  requirements.txt
```

---

## Quick Start

```bash
pip install -r requirements.txt

# Start first instance (port 8001)
python -m nwp_v12 --port 8001

# Start second instance (port 8002) for P2P testing
python -m nwp_v12 --port 8002

# Open http://127.0.0.1:8001 in your browser
```

Port must be in the range **8001-8020** (the auto-discovery scan range).

---

## Architecture

### Prediction Pipeline

1. **LSTM** (CPU, word-level, ~160 hidden units) produces candidate words with softmax confidence.
2. **Local n-gram** (trigram trie, capped at 1000 words) adds frequency-based candidates.
3. **Global n-gram** (aggregated from peer models via FL) adds peer-contributed candidates.
4. **Confidence-gated blend**: if LSTM max-confidence >= threshold, use the three-source blend; otherwise fall back to local + global n-gram only.
5. **Cache + personalisation**: session phrase cache and user dictionary re-rank results.

### Federated Learning

- **Gossip protocol**: peers exchange hybrid payloads (n-gram delta + gzip-compressed LSTM weights) at a configurable interval (default 5 min).
- **Delta sharing**: only the top-200 n-gram entries are shared per round to keep payloads small.
- **LSTM merging**: incoming peer weights are L2-normalised before blending to prevent dominant peers from overwriting local knowledge.
- **Peer discovery**: scans localhost ports 8001-8020 automatically every 60 s. Optional LAN scan available in FL Settings.

### Data

- All session text is run through `redact_text()` before storage (emails, URLs, IPs, long digit runs are replaced with placeholders).
- Model state is persisted to `data/trainer/` (SQLite DB + JSON model files).
- LSTM weights can be loaded from the training notebook (`NWP_Training_Notebook.ipynb`) via the **DL Tuning** tab in the UI.

---

## Configuration

All settings are persisted in the SQLite database and editable via the web UI:

| Setting | Default | Description |
|---|---|---|
| auto_train | true | Train model automatically after each session ends |
| auto_share | false | Push delta to peers automatically after training |
| gossip_enabled | true | Run gossip FL rounds at gossip_interval_s |
| gossip_interval_s | 300 | Seconds between gossip rounds |
| lstm_conf_threshold | 0.05 | Min LSTM softmax probability to use DL branch |
| lstm_weight | 0.50 | LSTM contribution in confident blend |
| local_ngram_weight | 0.20 | Local n-gram contribution in confident blend |
| global_ngram_weight | 0.30 | Peer n-gram contribution in confident blend |

---

## LSTM Weights

To use pre-trained LSTM weights:

1. Train using `NWP_Training_Notebook.ipynb` (Colab or local).
2. Export weights from the notebook (produces a `.json` file).
3. Upload via the **DL Tuning** tab in the web UI, or via:
   ```
   POST /local/model/upload_weights  (multipart: model_id + file)
   ```

The server runs fully without PyTorch installed -- it falls back to n-gram-only mode automatically.
