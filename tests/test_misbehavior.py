import unittest

from simple_chatbot.misbehavior import (
    KNOWN_MODES,
    Injection,
    MisbehaviorConfig,
    MisbehaviorPolicy,
)


class MisbehaviorConfigTests(unittest.TestCase):
    def test_rejects_rate_out_of_range(self):
        with self.assertRaises(ValueError):
            MisbehaviorConfig(rate=1.5, modes=("drop_retrieval",), seed=0)
        with self.assertRaises(ValueError):
            MisbehaviorConfig(rate=-0.1, modes=("drop_retrieval",), seed=0)

    def test_rejects_unknown_mode(self):
        with self.assertRaises(ValueError):
            MisbehaviorConfig(rate=0.5, modes=("not_a_mode",), seed=0)

    def test_accepts_known_modes(self):
        cfg = MisbehaviorConfig(rate=0.5, modes=tuple(sorted(KNOWN_MODES)), seed=0)
        self.assertEqual(cfg.rate, 0.5)


class MisbehaviorPolicyTests(unittest.TestCase):
    def _policy(self, rate, modes, seed=0):
        return MisbehaviorPolicy(MisbehaviorConfig(rate=rate, modes=tuple(modes), seed=seed))

    def test_rate_zero_never_fires(self):
        p = self._policy(0.0, ["drop_retrieval"])
        results = [p.maybe("retrieval", ["drop_retrieval"]) for _ in range(50)]
        self.assertTrue(all(r is None for r in results))
        self.assertEqual(p.injections, [])

    def test_rate_one_always_fires(self):
        p = self._policy(1.0, ["drop_retrieval"])
        results = [p.maybe("retrieval", ["drop_retrieval"]) for _ in range(50)]
        self.assertTrue(all(isinstance(r, Injection) for r in results))
        self.assertEqual(len(p.injections), 50)

    def test_only_enabled_eligible_modes_fire(self):
        p = self._policy(1.0, ["drop_retrieval"])  # poison not enabled
        inj = p.maybe("retrieval", ["poison_retrieval"])
        self.assertIsNone(inj)  # eligible but not enabled
        inj = p.maybe("retrieval", ["drop_retrieval", "poison_retrieval"])
        self.assertEqual(inj.mode, "drop_retrieval")  # only the enabled one can be chosen

    def test_same_seed_is_reproducible(self):
        modes = ["drop_retrieval", "poison_retrieval"]
        a = self._policy(0.5, modes, seed=7)
        b = self._policy(0.5, modes, seed=7)
        seq_a = [a.maybe("retrieval", modes) for _ in range(30)]
        seq_b = [b.maybe("retrieval", modes) for _ in range(30)]
        self.assertEqual(
            [None if i is None else i.mode for i in seq_a],
            [None if i is None else i.mode for i in seq_b],
        )

    def test_injection_records_stage_and_ctx(self):
        p = self._policy(1.0, ["ignore_retrieval"])
        inj = p.maybe("decision", ["ignore_retrieval"], {"turn": 2, "round": 1})
        self.assertEqual(inj.stage, "decision")
        self.assertEqual(inj.mode, "ignore_retrieval")
        self.assertEqual(inj.turn, 2)
        self.assertEqual(inj.round, 1)


if __name__ == "__main__":
    unittest.main()
