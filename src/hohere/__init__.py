"""hohere v2: a restartable Libri-Light dataset pipeline.

Part A (this package) is the non-ML half: remote tar indexing, chunking on the
Libri-Light VAD, the m-stream builder, record/shard writing, resume state and the
web status page. Part B adds the Mimi/WavLM models (`models` extra).
"""

__version__ = "2.0.0.dev0"
