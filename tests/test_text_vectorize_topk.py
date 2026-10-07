import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.feature_extraction.text import TfidfVectorizer as SkTfidf  # noqa: E402

from incident_engine.data import generate_incidents  # noqa: E402
from incident_engine.text import analyze, tokenize  # noqa: E402
from incident_engine.topk import topk_heap, topk_partition, topk_rows, topk_sort  # noqa: E402
from incident_engine.vectorize import ReferenceTfidf, SparseTfidf  # noqa: E402

DOCS = [i.text for i in generate_incidents(600, seed=3)]


class TextTests(unittest.TestCase):
    def test_tokenize_lowercases_and_splits_on_punctuation(self):
        self.assertEqual(tokenize("VPN-Gateway: Error NET-4821!"), ["vpn", "gateway", "error", "net", "4821"])

    def test_ngrams(self):
        self.assertEqual(analyze("a b c", (1, 2)), ["a", "b", "c", "a b", "b c"])
        self.assertEqual(analyze("a b c", (2, 2)), ["a b", "b c"])
        self.assertEqual(analyze("a", (1, 3)), ["a"])
        self.assertEqual(analyze("", (1, 2)), [])

    def test_invalid_range(self):
        with self.assertRaises(ValueError):
            analyze("a b", (2, 1))


class VectorizerTests(unittest.TestCase):
    def test_sparse_matches_reference_exactly(self):
        ref, sp = ReferenceTfidf(min_df=2), SparseTfidf(min_df=2)
        R = ref.fit_transform(DOCS)
        S = sp.fit_transform(DOCS).toarray()
        self.assertEqual(list(ref.terms_), list(sp.terms_))
        dense_ref = np.zeros_like(S)
        for r, vec in enumerate(R):
            for j, w in vec.items():
                dense_ref[r, j] = w
        np.testing.assert_allclose(S, dense_ref, atol=1e-12)

    def test_matches_scikit_learn(self):
        sk = SkTfidf(analyzer=lambda d: analyze(d, (1, 2)), min_df=2)
        expected = sk.fit_transform(DOCS)
        ours = SparseTfidf(min_df=2)
        got = ours.fit_transform(DOCS)
        self.assertEqual(list(sk.get_feature_names_out()), list(ours.terms_))
        np.testing.assert_allclose(got.toarray(), expected.toarray(), atol=1e-12)
        np.testing.assert_allclose(ours.idf_, sk.idf_, atol=1e-12)

    def test_transform_equals_fit_transform_on_training_docs(self):
        v = SparseTfidf(min_df=2)
        a = v.fit_transform(DOCS)
        b = v.transform(DOCS)
        np.testing.assert_allclose(a.toarray(), b.toarray(), atol=1e-12)

    def test_rows_are_unit_norm_or_empty(self):
        v = SparseTfidf(min_df=2).fit(DOCS)
        X = v.transform(DOCS[:50] + ["zzzzz qqqqq", ""])  # last two have no known terms
        norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
        np.testing.assert_allclose(norms[:50], 1.0, atol=1e-12)
        self.assertEqual(list(norms[50:]), [0.0, 0.0])

    def test_unseen_terms_ignored_and_shape_stable(self):
        v = SparseTfidf().fit(["alpha beta", "beta gamma"])
        X = v.transform(["alpha delta"])
        self.assertEqual(X.shape, (1, len(v.terms_)))
        self.assertEqual(X.nnz, 1)

    def test_min_df_prunes_rare_terms(self):
        v = SparseTfidf(ngram_range=(1, 1), min_df=2).fit(["a b", "a c", "a b"])
        self.assertEqual(list(v.terms_), ["a", "b"])

    def test_not_fitted(self):
        with self.assertRaises(RuntimeError):
            SparseTfidf().transform(["x"])


class TopKTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_three_implementations_agree_including_ties(self):
        for _ in range(60):
            n = int(self.rng.integers(1, 200))
            s = self.rng.choice([0.0, 0.1, 0.5, 0.9, 1.0], size=n) * (self.rng.random(n) < 0.7)  # lots of ties and zeros
            for k in (1, 3, 10, 500):
                a, b, c = topk_sort(s, k), topk_heap(s, k), topk_partition(s, k)
                np.testing.assert_array_equal(a, b)
                np.testing.assert_array_equal(a, c)

    def test_contract(self):
        s = np.array([0.2, 0.9, 0.9, 0.0, 0.5])
        np.testing.assert_array_equal(topk_partition(s, 3), [1, 2, 4])  # ties: lower index first
        np.testing.assert_array_equal(topk_partition(s, 10), [1, 2, 4, 0])  # zeros are never returned

    def test_edges(self):
        for f in (topk_sort, topk_heap, topk_partition):
            self.assertEqual(f(np.array([]), 3).size, 0)
            self.assertEqual(f(np.array([1.0, 2.0]), 0).size, 0)
            self.assertEqual(f(np.zeros(5), 2).size, 0)

    def test_topk_rows_padding(self):
        S = np.array([[0.0, 0.3, 0.1], [0.0, 0.0, 0.0]])
        idx, val = topk_rows(S, 2)
        np.testing.assert_array_equal(idx, [[1, 2], [-1, -1]])
        np.testing.assert_allclose(val, [[0.3, 0.1], [0.0, 0.0]])


if __name__ == "__main__":
    unittest.main()
