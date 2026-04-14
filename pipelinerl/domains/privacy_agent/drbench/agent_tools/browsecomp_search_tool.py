"""
BrowseComp-Plus search tool for DrBench.

Provides offline web search using a fixed FAISS-indexed corpus (BrowseComp-Plus).
Uses BM25 broad recall + dense re-ranking for hybrid retrieval.

The embedding model runs on CPU by default to avoid GPU memory conflicts with vLLM.
"""


import glob
import heapq
import logging
import math
import pickle
import re as re_mod
import threading
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from tevatron.retriever.searcher import FaissFlatSearcher
from tevatron.retriever.modeling.dense import DenseModel

from ..config import RunConfig
from ..internet_search_logging import log_internet_search
from .base import ResearchContext, Tool


# ---------------------------------------------------------------------------
# Inline BM25 (zero external deps, ~40 lines)
# ---------------------------------------------------------------------------
_TOKEN_RE = re_mod.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def _bm25_tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text) if len(t) >= 2]


class _BM25Index:
    """Minimal BM25 for BrowseComp recall stage."""

    def __init__(
        self,
        docs: Optional[list[str]] = None,
        k1: float = 1.5,
        b: float = 0.75,
        state: Optional["_BM25State"] = None,
    ):
        if state is not None:
            self._k1 = state.k1
            self._b = state.b
            self._N = state.N
            self._doc_len = list(state.doc_len)
            self._df = defaultdict(int, state.df)
            self._inv = defaultdict(list, state.inv)
            self._avgdl = state.avgdl
            return

        if not docs:
            raise ValueError("_BM25Index requires at least one document")

        self._k1, self._b = k1, b
        self._N = len(docs)
        self._doc_len: list[int] = []
        self._df: dict[str, int] = defaultdict(int)
        self._inv: dict[str, list[tuple[int, int]]] = defaultdict(list)
        total = 0
        for idx, text in enumerate(docs):
            tf = Counter(_bm25_tokenize(text))
            dl = sum(tf.values())
            self._doc_len.append(dl)
            total += dl
            for term, f in tf.items():
                self._df[term] += 1
                self._inv[term].append((idx, f))
        self._avgdl = (total / self._N) if self._N else 0.0

    def search(self, query: str, k: int = 200) -> list[tuple[int, float]]:
        """Return list of (doc_idx, score) sorted by BM25 score."""
        qtf = Counter(_bm25_tokenize(query))
        if not qtf:
            return []
        scores: dict[int, float] = defaultdict(float)
        for term in qtf:
            df = self._df.get(term)
            if not df:
                continue
            idf = math.log(1.0 + (self._N - df + 0.5) / (df + 0.5))
            for doc_idx, f in self._inv.get(term, []):
                dl = self._doc_len[doc_idx]
                denom = f + self._k1 * (1.0 - self._b + self._b * (dl / (self._avgdl or 1.0)))
                scores[doc_idx] += idf * (f * (self._k1 + 1.0) / denom)
        if not scores:
            return []
        return heapq.nlargest(k, scores.items(), key=lambda kv: kv[1])

    def to_state(self) -> "_BM25State":
        return _BM25State(
            k1=self._k1,
            b=self._b,
            N=self._N,
            doc_len=list(self._doc_len),
            avgdl=self._avgdl,
            df=dict(self._df),
            inv={term: list(postings) for term, postings in self._inv.items()},
        )

    @classmethod
    def from_state(cls, state: "_BM25State") -> "_BM25Index":
        return cls(state=state)


@dataclass(frozen=True)
class _BM25State:
    k1: float
    b: float
    N: int
    doc_len: list[int]
    avgdl: float
    df: dict[str, int]
    inv: dict[str, list[tuple[int, int]]]

logger = logging.getLogger(__name__)


class BrowseCompSearchTool(Tool):
    """Tool for searching the BrowseComp-Plus fixed corpus using dense retrieval.

    This tool provides reproducible, offline web search as an alternative to
    live Serper searches. Results come from a fixed corpus indexed with FAISS.

    The embedding model runs on CPU by default to avoid GPU memory conflicts
    with vLLM serving the main LLM.
    """

    def __init__(
        self,
        config: RunConfig,
        vector_store: Any = None,
        device: str = "cpu",
        helper_client: Any = None,
        use_remote: bool = False,
    ):
        """Initialize BrowseComp search tool.

        Args:
            config: RunConfig with BrowseComp settings (index_glob, model_name, etc.)
            vector_store: Optional vector store to store retrieved documents
            device: Device for embedding model ("cpu" by default to avoid vLLM conflicts)
        """
        self.config = config
        self.vector_store = vector_store
        self.device = device
        self.helper_client = helper_client
        self.use_remote = use_remote

        self.searcher: Optional[FaissFlatSearcher] = None
        self.lookup: Optional[List[str]] = None
        self.dataset = None
        self.docid_to_idx: Dict[str, int] = {}
        self.docid_to_url: Dict[str, str] = {}
        self.model = None
        self.tokenizer = None
        self.bm25: Optional[_BM25Index] = None
        # Maps BM25 internal index -> FAISS lookup index (for docid resolution)
        self._bm25_idx_to_lookup_idx: Optional[Dict[int, int]] = None

        # Task prefix for query encoding
        self.task_prefix = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"

        # Lock for thread-safe embedding model inference (shared across shallow copies)
        self._encode_lock = threading.Lock()

        if not self.use_remote:
            self._initialize()

    @property
    def purpose(self) -> str:
        # Match InternetSearchTool's purpose so agent treats it as regular web search
        return """External market research, competitive intelligence, and public data analysis.
        IDEAL FOR: Market trends, competitor analysis, industry reports, public research papers, news articles, regulatory information, and technology comparisons.
        USE WHEN: Research requires public/external sources, competitor benchmarking, market validation, industry context, or recent developments.
        PARAMETERS: query (specific search terms work best - e.g., 'AI market size 2024', 'competitor pricing strategies', 'regulatory changes fintech')
        OUTPUTS: Search results with URLs, snippets, and relevant content that gets automatically processed and stored for synthesis."""

    def _initialize(self) -> None:
        """Initialize FAISS index, BM25 index, embedding model, and corpus."""
        logger.info("Initializing BrowseComp-Plus search tool...")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Index: {self.config.browsecomp_index_glob}")
        logger.info(f"  Model: {self.config.browsecomp_model_name}")

        self._load_faiss_index()
        self._load_model_and_tokenizer()
        self._load_dataset()
        self._build_bm25()

        logger.info("BrowseComp-Plus search tool initialized.")

    def _load_faiss_index(self) -> None:
        """Load FAISS index from pickle shards."""
        def pickle_load(path: str):
            with open(path, "rb") as f:
                reps, lookup = pickle.load(f)
            return np.array(reps), lookup

        index_files = sorted(glob.glob(self.config.browsecomp_index_glob))
        if not index_files:
            raise RuntimeError(
                f"No index shards found for pattern: {self.config.browsecomp_index_glob}"
            )

        logger.info(f"Loading {len(index_files)} index shards...")
        reps0, lookup0 = pickle_load(index_files[0])
        self.searcher = FaissFlatSearcher(reps0)
        self.searcher.add(reps0)  # FaissFlatSearcher.__init__ only sets dimension
        self.lookup = list(lookup0)

        for path in index_files[1:]:
            reps, shard_lookup = pickle_load(path)
            self.searcher.add(reps)
            self.lookup.extend(shard_lookup)

        logger.info(f"Loaded index with {len(self.lookup)} documents.")

    def _load_model_and_tokenizer(self) -> None:
        """Load embedding model and tokenizer on specified device."""
        # Use float32 for CPU, float16 for GPU
        torch_dtype = torch.float16 if self.device.startswith("cuda") else torch.float32

        logger.info(f"Loading embedding model on {self.device}...")
        self.model = DenseModel.load(
            self.config.browsecomp_model_name,
            pooling="eos",
            normalize=True,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.browsecomp_model_name,
            padding_side="left",
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _load_dataset(self) -> None:
        """Load corpus dataset from HuggingFace."""
        dataset_source = self.config.browsecomp_corpus
        dataset_path = Path(dataset_source)

        if dataset_path.exists():
            logger.info(f"Loading local corpus file: {dataset_path}")
            self.dataset = load_dataset("json", data_files=str(dataset_path), split="train")
        else:
            logger.info(f"Loading dataset: {dataset_source}")
            self.dataset = load_dataset(dataset_source, split="train")

        first_row = self.dataset[0] if len(self.dataset) > 0 else {}
        if isinstance(first_row, dict) and "docid" not in first_row:
            flattened_docs: list[dict[str, str]] = []
            seen_docids: set[str] = set()
            for row in self.dataset:
                for field_name in ("gold_docs", "negative_docs", "evidence_docs"):
                    docs = row.get(field_name) or []
                    if not isinstance(docs, list):
                        continue
                    for doc in docs:
                        if not isinstance(doc, dict):
                            continue
                        docid = doc.get("docid")
                        if docid is None:
                            continue
                        docid_str = str(docid)
                        if docid_str in seen_docids:
                            continue
                        seen_docids.add(docid_str)
                        flattened_docs.append(
                            {
                                "docid": docid_str,
                                "url": doc.get("url") or "",
                                "text": doc.get("text") or "",
                            }
                        )
            self.dataset = flattened_docs

        for idx, row in enumerate(self.dataset):
            docid = row.get("docid")
            if docid is None:
                continue
            docid_str = str(docid)
            self.docid_to_idx[docid_str] = idx
            self.docid_to_idx[f"web/{docid_str}"] = idx
            self.docid_to_url[docid_str] = row.get("url") or ""
            self.docid_to_url[f"web/{docid_str}"] = row.get("url") or ""

        logger.info(f"Loaded {len(self.docid_to_idx)} documents from corpus.")

    def _build_bm25(self) -> None:
        """Build BM25 index over corpus texts for hybrid retrieval.

        Caches the index data as plain dicts to a pickle file next to the FAISS
        index shards so subsequent runs skip the expensive tokenization pass.
        """
        if not self.lookup:
            return

        index_dir = Path(glob.glob(self.config.browsecomp_index_glob)[0]).parent
        cache_path = index_dir / "bm25_cache.pkl"

        if cache_path.exists():
            logger.info(f"Loading cached BM25 index from {cache_path}")
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            state = _BM25State(
                k1=float(data["k1"]),
                b=float(data["b"]),
                N=int(data["N"]),
                doc_len=list(data["doc_len"]),
                avgdl=float(data["avgdl"]),
                df=dict(data["df"]),
                inv={term: list(postings) for term, postings in dict(data["inv"]).items()},
            )
            bm25 = _BM25Index.from_state(state)
            self.bm25 = bm25
            self._bm25_idx_to_lookup_idx = data["idx_to_lookup"]
            logger.info(f"BM25 index loaded ({bm25._N} documents).")
            return

        # Build lookup_docid -> FAISS-lookup-index mapping
        lookup_idx_by_docid: Dict[str, int] = {}
        for i, docid in enumerate(self.lookup):
            lookup_idx_by_docid[docid] = i

        bm25_texts: list[str] = []
        bm25_idx_to_lookup: Dict[int, int] = {}

        for docid, ds_idx in self.docid_to_idx.items():
            row = self.dataset[int(ds_idx)]
            text = row.get("text") or ""
            if not text:
                continue
            lookup_i = lookup_idx_by_docid.get(docid)
            if lookup_i is None:
                continue
            bm25_i = len(bm25_texts)
            bm25_texts.append(text)
            bm25_idx_to_lookup[bm25_i] = lookup_i

        logger.info(f"Building BM25 index over {len(bm25_texts)} documents...")
        self.bm25 = _BM25Index(bm25_texts)
        self._bm25_idx_to_lookup_idx = bm25_idx_to_lookup

        # Save raw data (no class instances) so any script can load it
        logger.info(f"Saving BM25 cache to {cache_path}")
        state = self.bm25.to_state()
        data = {
            "k1": state.k1,
            "b": state.b,
            "N": state.N,
            "doc_len": state.doc_len,
            "avgdl": state.avgdl,
            "df": state.df,
            "inv": state.inv,
            "idx_to_lookup": bm25_idx_to_lookup,
        }
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)
        logger.info("BM25 index built and cached.")

    def _encode_query(self, query: str) -> np.ndarray:
        """Encode query string to embedding vector (thread-safe)."""
        batch = self.tokenizer(
            self.task_prefix + query,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # Use autocast for GPU, no-op context for CPU
        if self.device.startswith("cuda"):
            ctx = torch.amp.autocast(device_type="cuda")
        else:
            ctx = nullcontext()

        with self._encode_lock:
            with ctx:
                with torch.no_grad():
                    reps = self.model.encode_query(batch)

        return reps.cpu().numpy()

    def _get_doc(self, docid: str) -> Dict[str, Any]:
        """Retrieve full document from corpus by docid."""
        idx = self.docid_to_idx.get(docid)
        if idx is None:
            return {"docid": docid, "url": "", "text": ""}

        row = self.dataset[int(idx)]
        return {
            "docid": docid,
            "url": row.get("url") or "",
            "text": row.get("text") or "",
        }

    def execute(self, query: str, context: ResearchContext) -> Dict[str, Any]:
        """Execute BrowseComp search and return results."""
        if self.use_remote:
            return self._execute_remote(query)

        if not self.searcher or not self.lookup:
            output = self.create_error_output(
                "browsecomp_search",
                query,
                "Searcher not initialized",
            )
            log_internet_search(
                self.config,
                tool="browsecomp_search",
                query=query,
                params={"top_k": self.config.browsecomp_top_k},
                result=output,
                extra={"source": "browsecomp"},
            )
            return output

    def _execute_remote(self, query: str) -> Dict[str, Any]:
        if self.helper_client is None:
            output = self.create_error_output(
                "browsecomp_search",
                query,
                "Remote helper client not configured",
            )
            log_internet_search(
                self.config,
                tool="browsecomp_search",
                query=query,
                params={"top_k": self.config.browsecomp_top_k},
                result=output,
                extra={"source": "browsecomp"},
            )
            return output

        try:
            results = self.helper_client.search_browsecomp(
                query=query,
                task_id=context.task_id,
                k=self.config.browsecomp_top_k,
                max_chars=self.config.browsecomp_max_chars,
            )
            return self._store_and_format_results(query=query, results=results)
        except Exception as exc:
            logger.error("Remote BrowseComp search failed: %s", exc)
            output = self.create_error_output(
                "browsecomp_search",
                query,
                f"Search failed: {exc}",
            )
            log_internet_search(
                self.config,
                tool="browsecomp_search",
                query=query,
                params={"top_k": self.config.browsecomp_top_k},
                result=output,
                extra={"source": "browsecomp"},
            )
            return output

    def _store_and_format_results(self, query: str, results: list[dict[str, Any]]) -> Dict[str, Any]:
        content_stored_count = 0
        if self.vector_store and results:
            for index, result in enumerate(results):
                if result.get("text"):
                    doc_id = self.vector_store.store_document(
                        content=result["text"],
                        metadata={
                            "type": "browsecomp_result",
                            "query": query,
                            "url": result.get("url", ""),
                            "docid": result.get("docid"),
                            "score": result.get("score"),
                            "search_rank": index + 1,
                            "timestamp": datetime.now().isoformat(),
                        },
                    )
                    if doc_id:
                        content_stored_count += 1

        output = self.create_success_output(
            tool_name="browsecomp_search",
            query=query,
            results=results,
            data_retrieved=len(results) > 0,
            results_count=len(results),
            source="browsecomp",
            content_stored_in_vector=content_stored_count,
            stored_in_vector=content_stored_count > 0,
        )

        log_internet_search(
            self.config,
            tool="browsecomp_search",
            query=query,
            params={"top_k": self.config.browsecomp_top_k},
            result=output,
            extra={"source": "browsecomp"},
        )
        return output

        try:
            top_k = self.config.browsecomp_top_k
            q_reps = self._encode_query(query)

            # Hybrid retrieval: BM25 broad recall + dense re-ranking.
            # Collect candidate FAISS-lookup indices from both BM25 and dense.
            candidate_lookup_idxs: set[int] = set()

            # Stage 1a: BM25 recall (200 candidates)
            if self.bm25 is not None and self._bm25_idx_to_lookup_idx is not None:
                bm25_hits = self.bm25.search(query, k=200)
                for bm25_idx, _score in bm25_hits:
                    lookup_i = self._bm25_idx_to_lookup_idx.get(bm25_idx)
                    if lookup_i is not None:
                        candidate_lookup_idxs.add(lookup_i)

            # Stage 1b: Dense recall (200 candidates)
            dense_scores, dense_indices = self.searcher.search(q_reps, min(200, len(self.lookup)))
            for idx in dense_indices[0]:
                candidate_lookup_idxs.add(int(idx))

            # Stage 2: Re-rank all candidates by inner product with query.
            # Embeddings are L2-normalized, so inner product == cosine similarity.
            candidate_list = sorted(candidate_lookup_idxs)
            candidate_embs = np.array([self.searcher.index.reconstruct(i) for i in candidate_list])
            scores_all = candidate_embs @ q_reps[0]

            ranked = sorted(zip(candidate_list, scores_all), key=lambda x: x[1], reverse=True)

            max_chars = self.config.browsecomp_max_chars
            results = []
            for lookup_idx, score in ranked[:top_k]:
                docid = self.lookup[lookup_idx]
                doc = self._get_doc(docid)
                text = doc.get("text") or ""
                if len(text) > max_chars:
                    text = text[:max_chars] + "\n\n[truncated]"
                results.append({
                    "docid": docid,
                    "score": float(score),
                    "url": doc.get("url"),
                    "text": text,
                })

            # Store in vector store if available
            content_stored_count = 0
            if self.vector_store and results:
                for i, result in enumerate(results):
                    if result.get("text"):
                        doc_id = self.vector_store.store_document(
                            content=result["text"],
                            metadata={
                                "type": "browsecomp_result",
                                "query": query,
                                "url": result.get("url", ""),
                                "docid": result.get("docid"),
                                "score": result.get("score"),
                                "search_rank": i + 1,
                                "timestamp": datetime.now().isoformat(),
                            },
                        )
                        if doc_id:
                            content_stored_count += 1

            output = self.create_success_output(
                tool_name="browsecomp_search",
                query=query,
                results=results,
                data_retrieved=len(results) > 0,
                results_count=len(results),
                source="browsecomp",
                content_stored_in_vector=content_stored_count,
                stored_in_vector=content_stored_count > 0,
            )

            log_internet_search(
                self.config,
                tool="browsecomp_search",
                query=query,
                params={"top_k": self.config.browsecomp_top_k},
                result=output,
                extra={"source": "browsecomp"},
            )
            return output

        except Exception as exc:
            logger.error(f"BrowseComp search failed: {exc}")
            output = self.create_error_output(
                "browsecomp_search",
                query,
                f"Search failed: {exc}",
            )
            log_internet_search(
                self.config,
                tool="browsecomp_search",
                query=query,
                params={"top_k": self.config.browsecomp_top_k},
                result=output,
                extra={"source": "browsecomp"},
            )
            return output
