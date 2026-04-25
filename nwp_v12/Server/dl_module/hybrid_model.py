"""
hybrid_model.py
LSTM-first next-word prediction with n-gram enhancement and confidence gating.

Architecture
------------
LstmWordModel    : Small CPU word-level LSTM with a safe no-torch fallback.
                   Falls back silently to returning ([], 0.0) when PyTorch is
                   not installed, allowing the application to run in n-gram-only
                   mode without any code-path changes in the caller.

HybridModelAgent : Combines local LSTM, local n-gram, and peer/global n-gram.
                   Prediction uses a confidence-gated lambda blend:

                   If LSTM max-confidence >= CONF_THRESHOLD:
                       score = LSTM_WEIGHT * DL + LOCAL_NGRAM_WEIGHT * local
                               + GLOBAL_NGRAM_WEIGHT * global

                   Else (LSTM unreliable -- n-gram fallback):
                       score = LOCAL_NGRAM_FALLBACK_WEIGHT * local
                               + GLOBAL_NGRAM_FALLBACK_WEIGHT * global

                   Weights in each branch sum to 1.0.

Key constants
-------------
LstmWordModel.CONF_THRESHOLD : Minimum softmax probability for the LSTM result
    to be included in the blend (default 0.05, tunable via NWP_LSTM_CONF_THRESHOLD
    env var or the DL Tuning tab in the web UI).  Raise to 0.15-0.25 after the
    model has accumulated substantial training data (>500 steps).

Federated sharing
-----------------
get_federated_payload(top_k)  : Serialise local n-gram delta + LSTM state.
apply_federated_payload(data) : Merge a peer payload into the global model.
    LSTM weights are L2-normalised before blending to prevent dominant peers
    from overwriting local knowledge when peers have different training depths.
"""
from __future__ import annotations

import base64
import gzip
import io
import json
import logging as _logging
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any

from .trie_model import NgramModel, build_seed_model, tokenize

_torch_import_error: str | None = None
try:
    import torch
    from torch import nn
except Exception as _e:
    torch = None  # type: ignore[assignment]
    nn = None     # type: ignore[assignment]
    _torch_import_error = str(_e)
    _logging.warning(
        "NWP: torch import failed -- running in n-gram-only mode. Error: %s", _e
    )

_WORD_RE = re.compile(r"[a-zA-Z']+|[0-9]+")


def _clean_word(word: str) -> str:
    """Extract the first clean alphanumeric token from word, lowercased."""
    m = _WORD_RE.search(word.lower())
    return m.group(0) if m else ""


# ---------------------------------------------------------------------------
# LSTM vocabulary (built once at import time from the seed corpus)
# ---------------------------------------------------------------------------

def _build_vocab(max_vocab: int = 2500) -> tuple[dict[str, int], list[str]]:
    """Build word <-> index mappings from the seed corpus."""
    from .trie_model import _load_seed_texts
    seed_texts = _load_seed_texts()
    words: dict[str, int] = {"<pad>": 0, "<unk>": 1, "<bos>": 2}
    for text in seed_texts:
        for word in tokenize(text):
            if word not in words and len(words) < max_vocab:
                words[word] = len(words)
    id_to_word = [""] * len(words)
    for word, idx in words.items():
        id_to_word[idx] = word
    return words, id_to_word


# ---------------------------------------------------------------------------
# LSTM Model Definition
# ---------------------------------------------------------------------------

class _WordLSTM(nn.Module if nn is not None else object):  # type: ignore[misc]
    """Two-layer embedding + LSTM + linear head for word-level LM."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 96,
        hidden_dim: int = 160,
    ) -> None:
        if nn is None:
            return
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tokens):  # type: ignore[override]
        x = self.embed(tokens)
        out, _ = self.lstm(x)
        return self.head(out)


@dataclass
class LstmPrediction:
    word: str
    score: float


# ---------------------------------------------------------------------------
# LSTM Word Model
# ---------------------------------------------------------------------------

class LstmWordModel:
    """
    Small CPU LSTM wrapper with safe no-torch fallback.

    When PyTorch is unavailable all methods return empty results or False so
    the rest of the application runs unchanged in n-gram-only mode.

    Warm-up
    -------
    A single dummy forward pass is run at init to pre-compile LSTM kernels and
    eliminate the cold-start latency on the first real prediction.

    Federated weight merging
    ------------------------
    apply_peer_state() normalises each incoming tensor by its L2 norm before
    the weighted blend.  This prevents peers with more training steps from
    dominating the merge and keeps aggregation numerically stable.
    """

    MAX_SEQ_LEN = 16
    TRAIN_STEPS = 10
    # Minimum softmax probability for the LSTM branch to be included in the
    # confidence-gated blend.  0.05 is a safe starting value; raise to
    # 0.15-0.25 after accumulating >500 real training steps.
    CONF_THRESHOLD = float(os.environ.get("NWP_LSTM_CONF_THRESHOLD", "0.05"))

    def __init__(self) -> None:
        self.word_to_id, self.id_to_word = _build_vocab()
        self.lock = threading.RLock()
        self.train_steps_done = 0
        self._init_torch()

    def _init_torch(self) -> None:
        """Attempt to (re-)initialise the LSTM model. Safe to call multiple times."""
        global torch, nn, _torch_import_error
        if torch is None:
            try:
                import torch as _torch
                from torch import nn as _nn
                torch = _torch
                nn = _nn
                _torch_import_error = None
                _logging.info("NWP: torch imported successfully on lazy re-attempt")
            except Exception as e:
                _torch_import_error = str(e)
                _logging.warning("NWP: torch re-import failed: %s", e)

        self.available = torch is not None and nn is not None
        if not self.available:
            self.model = None
            self.optimizer = None
            return

        self.model = _WordLSTM(len(self.id_to_word))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.003)
        self.model.eval()
        self._warmup()

    def reinitialise(self) -> dict[str, Any]:
        """Force a torch re-import and rebuild the model. Returns a status dict."""
        self._init_torch()
        if self.available and self.model is None:
            self.model = _WordLSTM(len(self.id_to_word))
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.003)
            self.model.eval()
            self._warmup()
        return {
            "available": self.available,
            "torch_error": _torch_import_error,
            "model_ready": self.model is not None,
        }

    def _warmup(self) -> None:
        """Run a dummy forward pass to pre-compile LSTM kernels."""
        if not self.available or self.model is None:
            return
        try:
            dummy = torch.zeros(1, 1, dtype=torch.long)
            with torch.no_grad():
                self.model(dummy)
        except Exception:
            pass

    def _ids(self, words: list[str]) -> list[int]:
        unk = self.word_to_id["<unk>"]
        return [self.word_to_id.get(w, unk) for w in words]

    def predict(
        self, context_words: list[str], k: int = 8
    ) -> tuple[list[LstmPrediction], float]:
        """
        Return (predictions, max_confidence).

        max_confidence is used by HybridModelAgent for confidence gating.
        Returns ([], 0.0) when PyTorch is unavailable.
        """
        if not self.available or self.model is None:
            return [], 0.0

        ids = self._ids(context_words[-self.MAX_SEQ_LEN:]) or [self.word_to_id["<bos>"]]
        x = torch.tensor([ids], dtype=torch.long)

        with self.lock, torch.no_grad():
            self.model.eval()
            logits = self.model(x)[0, -1].float()
            probs = torch.softmax(logits, dim=-1)
            top = torch.topk(probs, k=min(k + 8, len(self.id_to_word)))

        max_conf = float(top.values[0]) if len(top.values) > 0 else 0.0
        out: list[LstmPrediction] = []
        for idx, score in zip(top.indices.tolist(), top.values.tolist()):
            word = self.id_to_word[int(idx)]
            if word.startswith("<"):
                continue
            clean = _clean_word(word)
            if not clean:
                continue
            out.append(LstmPrediction(word=clean, score=float(score)))
            if len(out) >= k:
                break

        return out, max_conf

    def train_texts(self, texts: list[str]) -> int:
        """Fine-tune on a list of text strings. Returns the number of gradient steps."""
        if not self.available or self.model is None or self.optimizer is None:
            return 0

        sequences: list[list[int]] = []
        for text in texts:
            words = tokenize(text)
            if len(words) >= 2:
                ids = [self.word_to_id["<bos>"]] + self._ids(words)
                sequences.append(ids[-(self.MAX_SEQ_LEN + 1):])

        if not sequences:
            return 0

        loss_fn = nn.CrossEntropyLoss()
        steps = 0
        with self.lock:
            self.model.train()
            for seq in sequences[: self.TRAIN_STEPS]:
                if len(seq) < 2:
                    continue
                x = torch.tensor([seq[:-1]], dtype=torch.long)
                y = torch.tensor(seq[1:], dtype=torch.long)
                logits = self.model(x)[0]
                loss = loss_fn(logits, y)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # Gradient clipping prevents exploding gradients after FL updates.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                steps += 1
            self.model.eval()

        self.train_steps_done += steps
        return steps

    def state_payload(self) -> dict[str, Any] | None:
        """Serialise model weights as gzip+base64 JSON for federated sharing."""
        if not self.available or self.model is None:
            return None
        with self.lock:
            buf = io.BytesIO()
            torch.save(self.model.state_dict(), buf)
        raw = gzip.compress(buf.getvalue())
        return {
            "format": "torch_state_gzip_b64",
            "arch": "word_lstm_v1",
            "train_steps": self.train_steps_done,
            "blob": base64.b64encode(raw).decode("ascii"),
        }

    def apply_peer_state(self, payload: dict[str, Any], mix: float = 0.35) -> bool:
        """
        Merge peer weights into the local model.

        Each peer tensor is L2-normalised to match the magnitude of the local
        tensor before the weighted blend, preventing a heavily-trained peer
        from overwriting local knowledge entirely.

        Parameters
        ----------
        payload : Dict with 'arch', 'blob' (gzip+base64 state dict), and
                  'train_steps'.
        mix     : Weight given to the peer model (0 = no change, 1 = replace).
        """
        if not self.available or self.model is None or not payload:
            return False
        if payload.get("arch") != "word_lstm_v1":
            return False
        try:
            raw = gzip.decompress(base64.b64decode(payload.get("blob", "")))
            peer_state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
            with self.lock, torch.no_grad():
                own_state = self.model.state_dict()
                merged: dict[str, Any] = {}
                for key, own in own_state.items():
                    peer = peer_state.get(key)
                    if torch.is_tensor(peer) and peer.shape == own.shape:
                        peer_norm = peer.float().norm()
                        peer_normalised = (
                            peer.float() / (peer_norm + 1e-8) * own.float().norm()
                        )
                        merged[key] = (1.0 - mix) * own.float() + mix * peer_normalised
                    else:
                        merged[key] = own
                self.model.load_state_dict(merged, strict=True)
            return True
        except Exception:
            return False

    def save(self, path: str) -> None:
        """Persist weights to a JSON file."""
        if not self.available or self.model is None:
            return
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = self.state_payload()
        if payload:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f)

    def load(self, path: str) -> None:
        """Load weights from a JSON file produced by save() or the training notebook."""
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.apply_peer_state(payload, mix=1.0)
            self.train_steps_done = int(payload.get("train_steps", 0))
        except Exception:
            pass

    def load_external_weights(self, path: str) -> bool:
        """
        Load externally trained weights (e.g. from the training notebook).

        Expected format: JSON with the same schema as state_payload() output --
        keys 'arch', 'blob' (gzip+base64), and 'train_steps'.

        Returns True on success, False if the file is missing or incompatible.
        """
        if not os.path.exists(path):
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if payload.get("arch") != "word_lstm_v1":
                return False
            return self.apply_peer_state(payload, mix=1.0)
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Hybrid Model Agent
# ---------------------------------------------------------------------------

class HybridModelAgent:
    """
    Combines local LSTM, local n-gram, and peer/global n-gram into one predictor.

    Scoring (confidence-gated lambda blend)
    ----------------------------------------
    When LSTM max-confidence >= CONF_THRESHOLD (confident branch):
        score = LSTM_WEIGHT * DL_prob
                + LOCAL_NGRAM_WEIGHT * local_ngram
                + GLOBAL_NGRAM_WEIGHT * global_ngram

    When LSTM max-confidence < CONF_THRESHOLD (fallback branch):
        score = LOCAL_NGRAM_FALLBACK_WEIGHT * local_ngram
                + GLOBAL_NGRAM_FALLBACK_WEIGHT * global_ngram

    All per-source scores are max-normalised before blending so no single
    source dominates due to scale differences.
    """

    # Confident branch weights (sum to 1.0).
    LSTM_WEIGHT = 0.50
    LOCAL_NGRAM_WEIGHT = 0.20
    GLOBAL_NGRAM_WEIGHT = 0.30

    # Fallback branch weights (sum to 1.0).
    LOCAL_NGRAM_FALLBACK_WEIGHT = 0.40
    GLOBAL_NGRAM_FALLBACK_WEIGHT = 0.60

    def __init__(self) -> None:
        self.lstm = LstmWordModel()
        self.local: NgramModel = build_seed_model()
        self.global_: NgramModel = build_seed_model()
        self.local_version = 0
        self.global_version = 0
        self.weight_version = 0

    @property
    def engine(self) -> str:
        return "lstm+ngram" if self.lstm.available else "ngram-fallback"

    def observe(self, text: str, *, train_lstm: bool = True) -> int:
        """Update the local n-gram model and optionally fine-tune the LSTM."""
        words = tokenize(text)
        if not words:
            return 0
        self.local.observe(words)
        self.local_version += 1
        return self.lstm.train_texts([text]) if train_lstm else 0

    def train_texts(self, texts: list[str]) -> int:
        """Batch-train on a list of text strings. Returns total words observed."""
        total_words = 0
        trainable: list[str] = []
        for text in texts:
            words = tokenize(text)
            if words:
                self.local.observe(words)
                total_words += len(words)
                trainable.append(text)
        if total_words:
            self.local_version += 1
        self.lstm.train_texts(trainable)
        return total_words

    def predict(
        self, context_text: str, cursor_pos: int, k: int = 5
    ) -> tuple[list[dict[str, Any]], float]:
        """
        Return (suggestions, latency_ms).

        Each suggestion dict contains:
            text, score, source, mode, partial_word, lstm_conf.

        Mode is 'autocomplete' when the cursor is mid-word (no trailing space),
        otherwise 'next_word'.  Autocomplete uses n-gram trie prefix search only
        because the LSTM operates at the word boundary level.

        Note: suggestion text has no trailing space -- the frontend
        acceptSuggestion handler injects the space after insertion.
        """
        t0 = time.monotonic()
        prefix = context_text[:cursor_pos]
        words = tokenize(prefix)
        partial_word = ""

        if prefix and re.search(r"\w$", prefix):
            partial_word = words[-1] if words else ""
            context_words = words[:-1]
            mode = "autocomplete"
        else:
            context_words = words
            mode = "next_word"

        lstm_raw: dict[str, float] = {}
        local_raw: dict[str, float] = {}
        global_raw: dict[str, float] = {}
        lstm_conf = 0.0

        if mode == "autocomplete" and partial_word:
            for word, score in self.local.autocomplete(partial_word, k=k * 4):
                local_raw[_clean_word(word)] = score
            for word, score in self.global_.autocomplete(partial_word, k=k * 4):
                global_raw[_clean_word(word)] = score
        else:
            lstm_preds, lstm_conf = self.lstm.predict(context_words, k=k * 4)
            for pred in lstm_preds:
                lstm_raw[pred.word] = pred.score
            for word, score in self.local.predict(context_words, k=k * 4):
                local_raw[_clean_word(word)] = score
            for word, score in self.global_.predict(context_words, k=k * 4):
                global_raw[_clean_word(word)] = score

        def _norm(d: dict[str, float]) -> dict[str, float]:
            if not d:
                return d
            mx = max(d.values())
            return {w: v / mx for w, v in d.items()} if mx > 0 else d

        lstm_n = _norm(lstm_raw)
        local_n = _norm(local_raw)
        global_n = _norm(global_raw)
        all_words = set(lstm_n) | set(local_n) | set(global_n)

        use_lstm = lstm_conf >= LstmWordModel.CONF_THRESHOLD
        scores: dict[str, tuple[float, str]] = {}

        for word in all_words:
            if not word:
                continue
            if use_lstm:
                contributions = [
                    (lstm_n.get(word, 0.0) * self.LSTM_WEIGHT, "lstm"),
                    (local_n.get(word, 0.0) * self.LOCAL_NGRAM_WEIGHT, "ngram"),
                    (global_n.get(word, 0.0) * self.GLOBAL_NGRAM_WEIGHT, "peer-ngram"),
                ]
            else:
                contributions = [
                    (local_n.get(word, 0.0) * self.LOCAL_NGRAM_FALLBACK_WEIGHT, "ngram"),
                    (global_n.get(word, 0.0) * self.GLOBAL_NGRAM_FALLBACK_WEIGHT, "peer-ngram"),
                ]
            score = sum(c for c, _ in contributions)
            source = max(contributions, key=lambda x: x[0])[1]
            scores[word] = (score, source)

        ranked = sorted(scores.items(), key=lambda item: item[1][0], reverse=True)
        latency_ms = (time.monotonic() - t0) * 1000

        return [
            {
                "text": word,
                "score": round(score, 6),
                "source": source,
                "mode": mode,
                "partial_word": partial_word,
                "lstm_conf": round(lstm_conf, 4),
            }
            for word, (score, source) in ranked[:k]
        ], round(latency_ms, 2)

    def update_global(self, peer_model: NgramModel) -> None:
        """Merge a peer n-gram model into the global model."""
        self.global_.merge(peer_model)
        self.global_version += 1

    def apply_federated_payload(self, payload: dict[str, Any]) -> bool:
        """Merge a peer's hybrid payload (n-gram delta + optional LSTM state)."""
        ngram_data = payload.get("ngram") or payload
        if ngram_data:
            peer_model = NgramModel()
            peer_model.from_dict(ngram_data)
            self.update_global(peer_model)
        if self.lstm.apply_peer_state(payload.get("lstm_state") or {}):
            self.weight_version += 1
        return True

    def get_local_delta(self, top_k: int = 200) -> dict[str, Any]:
        """Extract top-k n-grams from the local model as a compact sharing payload."""
        data = self.local.to_dict()
        ngrams = data.get("ngrams", {})
        pruned: dict[str, dict[str, int]] = {}
        for ctx, word_counts in ngrams.items():
            top = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
            if top:
                pruned[ctx] = dict(top)
        unigram_top = sorted(
            data["unigram"].items(), key=lambda x: x[1], reverse=True
        )[:top_k]
        return {
            "version": 1,
            "max_ngram": self.local.max_ngram,
            "unigram": dict(unigram_top),
            "total_unigrams": sum(c for _, c in unigram_top),
            "ngrams": pruned,
        }

    def get_federated_payload(self, top_k: int = 200) -> dict[str, Any]:
        """Build the full hybrid payload for federated sharing."""
        return {
            "version": 2,
            "kind": "hybrid_lstm_ngram",
            "engine": self.engine,
            "ngram": self.get_local_delta(top_k=top_k),
            "lstm_state": self.lstm.state_payload(),
        }

    def count_entries(self) -> int:
        return self.local.count_entries()

    def to_persistence_dict(self) -> dict[str, Any]:
        """Serialise full model state for on-disk persistence."""
        ld = self.local.to_dict()
        ld["local_version"] = self.local_version
        gd = self.global_.to_dict()
        gd["global_version"] = self.global_version
        return {
            "local": ld,
            "global": gd,
            "engine": self.engine,
            "weight_version": self.weight_version,
        }

    def from_persistence_dict(self, data: dict[str, Any]) -> None:
        """Restore model state from a dict produced by to_persistence_dict()."""
        if "local" in data:
            self.local = NgramModel()
            self.local.from_dict(data["local"])
            self.local_version = int(
                data["local"].get("local_version", self.local_version)
            )
        if "global" in data:
            self.global_ = NgramModel()
            self.global_.from_dict(data["global"])
            self.global_version = int(
                data["global"].get("global_version", self.global_version)
            )
        self.weight_version = int(data.get("weight_version", 0))
