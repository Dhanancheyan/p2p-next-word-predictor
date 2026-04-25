"""
dl_module package -- prediction engines for NWP v11.

Modules
-------
trie_model      : NgramModel (trie-based trigram/bigram/unigram LM) and
                  build_seed_model() factory.
hybrid_model    : HybridModelAgent -- confidence-gated LSTM + n-gram blend.
                  LstmWordModel -- small CPU LSTM with safe no-torch fallback.
cache_agent     : CacheAgent -- session-level phrase cache (<0.1 ms lookups).
personalization : PersonalizationLayer -- recency boost + user dictionary.
redact          : redact_text() -- lightweight PII scrubber.
hashing         : sha256_hex() -- deterministic text hashing utility.
"""
