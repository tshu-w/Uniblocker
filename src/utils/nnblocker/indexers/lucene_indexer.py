import itertools
import shlex
from pathlib import Path
from runpy import run_module
from unittest.mock import patch

from pyserini.analysis import JWhiteSpaceAnalyzer
from pyserini.pyclass import autoclass
from pyserini.search import LuceneSearcher

from .indexer import BatchSearchResult, Indexer

BooleanQuery = autoclass("org.apache.lucene.search.BooleanQuery")
Integer = autoclass("java.lang.Integer")
BooleanQuery.setMaxClauseCount(Integer.MAX_VALUE)


class LuceneIndexer(Indexer):
    def __init__(
        self,
        save_dir: str,
        index_argv: str = ["--keepStopwords", "--stemmer", "none", "--pretokenized"],
        threads: int = 1,
    ):
        self.save_dir = Path(save_dir)
        self.index_argv = index_argv
        self.threads = threads

    def build_index(self, data):
        data = data.apply(" ".join)
        data.reset_index(inplace=True, drop=True)
        data = data.to_frame("contents")
        data.index.names = ["id"]
        data = data.reset_index()
        data["id"] = data["id"].astype("str")
        corpus_file = self.save_dir / "corpus.jsonl"
        data.to_json(corpus_file, orient="records", lines=True, force_ascii=False)

        index_dir = self.save_dir / "lucene"
        argv = list(
            itertools.chain(
                ["pyserini.index.lucene"],
                ["--collection", "JsonCollection"],
                ["--input", str(self.save_dir)],
                ["--index", str(index_dir)],
                ["--threads", str(self.threads)],
                self.index_argv,
            )
        )
        print(shlex.join(argv))
        with patch("sys.argv", argv):
            run_module(argv[0], run_name="__main__")

        self._searcher = LuceneSearcher(str(index_dir))
        analyzer = JWhiteSpaceAnalyzer()
        self._searcher.set_analyzer(analyzer)

    def batch_search(self, queries, k: int = 10) -> BatchSearchResult:
        queries = queries.apply(" ".join).to_list()
        query_ids = list(map(str, range(len(queries))))
        results = self._searcher.batch_search(queries, query_ids, k=k)
        values = [results[str(i)] for i in range(len(queries))]
        scores = [[r.score for r in rl] for rl in values]
        indices = [[int(r.docid) for r in rl] for rl in values]
        return BatchSearchResult(scores, indices)
