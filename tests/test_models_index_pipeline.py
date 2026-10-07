import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.naive_bayes import MultinomialNB  # noqa: E402

from incident_engine import PipelineConfig, TriagePipeline, generate_incidents  # noqa: E402
from incident_engine import models  # noqa: E402
from incident_engine.index import LSHIndex, ReferenceIndex, SparseExactIndex, create_index  # noqa: E402
from incident_engine.metrics import accuracy, macro_f1, overlap_at_k, retrieval_scores, score_ratio_at_k  # noqa: E402
from incident_engine.models import MultinomialNaiveBayes, NotFittedError, SoftmaxRegression  # noqa: E402
from incident_engine.vectorize import ReferenceTfidf, SparseTfidf  # noqa: E402

INC = generate_incidents(1500, seed=5)
TEXTS = [i.text for i in INC]
CATS = [i.category for i in INC]


class DataTests(unittest.TestCase):
    def test_deterministic_and_problem_consistent(self):
        self.assertEqual(generate_incidents(300, seed=1), generate_incidents(300, seed=1))
        self.assertNotEqual(generate_incidents(300, seed=1), generate_incidents(300, seed=2))
        by_problem = {}
        for i in INC:
            by_problem.setdefault(i.problem_id, set()).add(i.category)
        self.assertTrue(all(len(c) == 1 for c in by_problem.values()))  # a problem has one category

    def test_no_empty_tickets(self):
        self.assertTrue(all(i.text.strip() for i in generate_incidents(5000, seed=11)))

    def test_all_categories_and_priorities_present(self):
        self.assertEqual(len({i.category for i in INC}), 7)
        self.assertEqual({i.priority for i in INC}, {"P1", "P2", "P3", "P4"})


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vec = SparseTfidf(min_df=2)
        cls.X = cls.vec.fit_transform(TEXTS)

    def test_naive_bayes_matches_scikit_learn(self):
        ours = MultinomialNaiveBayes(alpha=0.5).fit(self.X, CATS)
        sk = MultinomialNB(alpha=0.5).fit(self.X, CATS)
        np.testing.assert_array_equal(ours.classes_, sk.classes_)
        np.testing.assert_allclose(ours.predict_log_proba(self.X), sk.predict_log_proba(self.X), atol=1e-9)
        np.testing.assert_array_equal(ours.predict(self.X), sk.predict(self.X))

    def test_softmax_gradient_matches_finite_differences(self):
        rng = np.random.default_rng(0)
        X = sparse.random(12, 6, density=0.6, format="csr", random_state=1)
        y = rng.integers(0, 3, size=12)
        W, b = rng.normal(size=(6, 3)) * 0.3, rng.normal(size=3) * 0.1
        _, gW, gb = SoftmaxRegression.loss_and_grad(W, b, X, y, l2=0.01)
        eps = 1e-6
        for (i, j) in [(0, 0), (2, 1), (5, 2)]:
            Wp, Wm = W.copy(), W.copy()
            Wp[i, j] += eps
            Wm[i, j] -= eps
            num = (SoftmaxRegression.loss_and_grad(Wp, b, X, y, 0.01)[0] - SoftmaxRegression.loss_and_grad(Wm, b, X, y, 0.01)[0]) / (2 * eps)
            self.assertAlmostEqual(gW[i, j], num, places=6)
        for j in range(3):
            bp, bm = b.copy(), b.copy()
            bp[j] += eps
            bm[j] -= eps
            num = (SoftmaxRegression.loss_and_grad(W, bp, X, y, 0.01)[0] - SoftmaxRegression.loss_and_grad(W, bm, X, y, 0.01)[0]) / (2 * eps)
            self.assertAlmostEqual(gb[j], num, places=6)

    def test_softmax_learns_and_loss_decreases(self):
        m = SoftmaxRegression(epochs=8, seed=0).fit(self.X, CATS)
        self.assertGreater(accuracy(CATS, m.predict(self.X)), 0.9)
        self.assertLess(m.loss_history_[-1], m.loss_history_[0])

    def test_softmax_is_deterministic_for_a_seed(self):
        a = SoftmaxRegression(epochs=2, seed=3).fit(self.X, CATS).coef_
        b = SoftmaxRegression(epochs=2, seed=3).fit(self.X, CATS).coef_
        np.testing.assert_array_equal(a, b)

    def test_probabilities_sum_to_one(self):
        for name in models.available():
            m = models.create(name).fit(self.X, CATS)
            np.testing.assert_allclose(m.predict_proba(self.X[:20]).sum(axis=1), 1.0, atol=1e-9)

    def test_registry_and_validation(self):
        self.assertEqual(models.available(), ["naive_bayes", "softmax"])
        with self.assertRaises(ValueError):
            models.create("random_forest")
        with self.assertRaises(NotFittedError):
            MultinomialNaiveBayes().predict(self.X)
        with self.assertRaises(ValueError):
            MultinomialNaiveBayes().fit(self.X, CATS[:-1])
        with self.assertRaises(ValueError):
            MultinomialNaiveBayes().fit(self.X, ["only"] * len(CATS))
        with self.assertRaises(ValueError):
            MultinomialNaiveBayes(alpha=0)
        fitted = MultinomialNaiveBayes().fit(self.X, CATS)
        with self.assertRaises(ValueError):
            fitted.predict(sparse.csr_matrix((1, self.X.shape[1] + 1)))

    def test_explain_returns_terms_present_in_document(self):
        m = SoftmaxRegression(epochs=5).fit(self.X, CATS)
        row = self.X[0]
        terms = m.explain(row, m.predict(row)[0], self.vec.terms_, top=3)
        present = {self.vec.terms_[j] for j in row.indices}
        self.assertTrue(terms and all(t in present for t, _ in terms))


class IndexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vec = SparseTfidf(min_df=2)
        cls.X = cls.vec.fit_transform(TEXTS)
        cls.ref_vec = ReferenceTfidf(min_df=2)
        cls.R = cls.ref_vec.fit_transform(TEXTS)
        cls.q = slice(0, 40)

    def test_sparse_exact_equals_reference(self):
        ref = ReferenceIndex().build(self.R)
        fast = SparseExactIndex().build(self.X)
        ri, rv = ref.query(self.R[self.q], 5)
        fi, fv = fast.query(self.X[self.q], 5)
        np.testing.assert_allclose(fv, rv, atol=1e-12)  # scores identical
        # indices identical wherever scores are not tied
        for r in range(rv.shape[0]):
            for c in range(5):
                tied = np.isclose(rv[r], rv[r, c], atol=1e-12).sum() > 1
                if not tied:
                    self.assertEqual(fi[r, c], ri[r, c])

    def test_chunking_does_not_change_results(self):
        big = SparseExactIndex().build(self.X).query(self.X[self.q], 5)
        tiny = SparseExactIndex(max_block_bytes=8 * self.X.shape[0] * 3).build(self.X).query(self.X[self.q], 5)  # 3 queries/chunk
        np.testing.assert_array_equal(big[0], tiny[0])
        np.testing.assert_allclose(big[1], tiny[1])

    def test_a_document_scores_one_against_itself(self):
        idx, val = SparseExactIndex().build(self.X).query(self.X[10], 3)
        self.assertAlmostEqual(val[0, 0], 1.0, places=9)
        self.assertTrue(np.isclose(val[0], 1.0).any())

    def test_no_overlap_returns_padding(self):
        idx, val = SparseExactIndex().build(self.X).query(self.vec.transform(["zzzz qqqq"]), 3)
        self.assertEqual(list(idx[0]), [-1, -1, -1])
        self.assertEqual(list(val[0]), [0.0, 0.0, 0.0])

    def test_lsh_scores_are_exact_and_recall_is_high(self):
        exact = SparseExactIndex().build(self.X)
        lsh = LSHIndex(seed=0).build(self.X)
        ei, ev = exact.query(self.X[:100], 5)
        li, lv = lsh.query(self.X[:100], 5)
        self.assertGreater(score_ratio_at_k(lv, ev), 0.9)
        self.assertGreater(overlap_at_k(li, ei), 0.6)
        self.assertLess(lsh.last_candidate_fraction, 0.6)  # it really scores only a fraction of the corpus
        for r in range(100):  # any returned doc must carry its exact cosine score
            for j, s in zip(li[r], lv[r]):
                if j >= 0:
                    self.assertAlmostEqual(s, float(self.X[r].multiply(self.X[j]).sum()), places=9)

    def test_validation_and_factory(self):
        with self.assertRaises(ValueError):
            SparseExactIndex().build(self.X).query(self.X[:1], 0)
        with self.assertRaises(ValueError):
            SparseExactIndex().build(self.X).query(sparse.csr_matrix((1, 3)), 3)
        with self.assertRaises(ValueError):
            LSHIndex(n_bits=0)
        self.assertIsInstance(create_index("lsh", n_bits=6), LSHIndex)
        with self.assertRaises(ValueError):
            create_index("hnsw")


class MetricTests(unittest.TestCase):
    def test_accuracy_and_f1(self):
        self.assertEqual(accuracy([1, 1, 0, 0], [1, 0, 0, 0]), 0.75)
        self.assertAlmostEqual(macro_f1([1, 1, 0, 0], [1, 0, 0, 0]), (0.8 + 2 / 3) / 2)
        with self.assertRaises(ValueError):
            accuracy([], [])

    def test_retrieval_scores(self):
        got = np.array([[5, 7, 9], [1, 2, 3], [-1, -1, -1]])
        res = retrieval_scores(got, np.array([7, 9, 4]), 3)
        self.assertAlmostEqual(res["hit@3"], 1 / 3)
        self.assertAlmostEqual(res["mrr"], (0.5 + 0 + 0) / 3)

    def test_overlap_at_k(self):
        self.assertEqual(overlap_at_k(np.array([[1, 2, -1]]), np.array([[1, 2, 3]])), 2 / 3)

    def test_score_ratio_ignores_ties_and_empty_queries(self):
        exact, approx = np.array([[1.0, 1.0], [0.0, 0.0]]), np.array([[1.0, 0.5], [0.0, 0.0]])
        self.assertAlmostEqual(score_ratio_at_k(approx, exact), 0.75)  # the all-zero query is skipped


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipe = TriagePipeline().fit(INC[:1200])

    def test_triage_end_to_end(self):
        res = self.pipe.triage("vpn gateway connection timed out error NET-1234 all users affected", k=3)
        self.assertEqual(res.category, "network")
        self.assertTrue(0 < res.category_confidence <= 1 and res.priority in {"P1", "P2", "P3", "P4"})
        self.assertTrue(res.similar and res.similar[0].similarity >= res.similar[-1].similarity)
        self.assertTrue(res.top_terms)

    def test_batched_equals_single(self):
        texts = [i.text for i in INC[1200:1210]]
        batch = self.pipe.triage_many(texts, 3)
        for t, b in zip(texts, batch):
            self.assertEqual(self.pipe.triage(t, 3), b)

    def test_beats_majority_baseline_on_held_out(self):
        test = INC[1200:]
        pred = [r.category for r in self.pipe.triage_many([i.text for i in test], 1)]
        majority = max(set(CATS), key=CATS.count)
        self.assertGreater(accuracy([i.category for i in test], pred), accuracy([i.category for i in test], [majority] * len(test)) + 0.5)

    def test_validation(self):
        with self.assertRaises(ValueError):
            self.pipe.triage("   ")
        with self.assertRaises(NotFittedError):
            TriagePipeline().triage("x")
        with self.assertRaises(ValueError):
            TriagePipeline().fit([])

    def test_algorithms_are_swappable_by_config(self):
        cfg = PipelineConfig(category_model="naive_bayes", index="lsh", index_params={"n_bits": 8, "n_tables": 8})
        p = TriagePipeline(cfg).fit(INC[:1000])
        held_out = INC[1000:1200]  # an out-of-vocabulary hand-written ticket is a bad unit test; use held-out data
        acc = np.mean([p.triage(i.text).category == i.category for i in held_out])
        self.assertGreater(acc, 0.85)
        self.assertIsInstance(p.index, LSHIndex)


@unittest.skipUnless(importlib.util.find_spec("fastapi") and importlib.util.find_spec("httpx"), "fastapi/httpx not installed")
class ApiTests(unittest.TestCase):
    def test_triage_endpoint(self):
        from fastapi.testclient import TestClient
        from incident_engine.api import create_app
        c = TestClient(create_app(n_train=1500))
        self.assertEqual(c.get("/health").status_code, 200)
        r = c.post("/triage", json={"text": "dns resolver high latency error NET-2000", "k": 3})
        self.assertEqual(r.status_code, 200)
        self.assertIn("category", r.json())
        self.assertEqual(c.post("/triage", json={"text": ""}).status_code, 422)


if __name__ == "__main__":
    unittest.main()
