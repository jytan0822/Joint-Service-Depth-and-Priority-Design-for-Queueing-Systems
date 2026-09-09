"""Read-only regression checks for the inherited empirical calibration."""
import unittest

import numpy as np

from sgd_calibration import (
    bootstrap_calibrations, calibration_from_indices, calibration_tables,
    load_calibration,
)


class CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cal = load_calibration()

    def test_recorded_strict_panel(self):
        self.assertEqual(self.cal['outcomes'].shape, (200, 5))
        self.assertEqual(len(np.unique(self.cal['dialogue_ids'])), 197)
        np.testing.assert_array_equal(self.cal['outcomes'].sum(axis=0), [77, 86, 90, 92, 93])
        self.assertEqual(len(set(self.cal['case_ids'])), 200)

    def test_cost_scale_and_token_accounting(self):
        c = self.cal
        self.assertAlmostEqual(c['times'].mean(), 1.0, places=13)
        np.testing.assert_allclose(c['times'], c['costs'] / c['costs'].mean())
        np.testing.assert_allclose(c['costs'], c['input_costs'] + c['output_costs'])
        np.testing.assert_array_equal(c['output_tokens'], c['reasoning_tokens'] + c['visible_output_tokens'])
        np.testing.assert_array_equal(c['total_tokens'], c['input_tokens'] + c['output_tokens'])

    def test_bootstrap_preserves_entire_dialogues_and_effort_pairs(self):
        c = self.cal
        boot = bootstrap_calibrations(c, n_boot=16, seed=123)
        clusters, inverse = np.unique(c['dialogue_ids'], return_inverse=True)
        self.assertEqual(len(clusters), 197)
        for j, counts in enumerate(boot['cluster_counts']):
            indices = np.repeat(np.arange(len(inverse)), counts[inverse])
            explicit = calibration_from_indices(c, indices)
            self.assertEqual(len(indices), boot['draw_size'][j])
            for key in ('values', 'means', 'seconds'):
                np.testing.assert_allclose(boot[key][j], explicit[key], rtol=1e-12)
        np.testing.assert_allclose(boot['means'].mean(axis=1), 1.0)

    def test_chunking_does_not_change_resamples(self):
        a = bootstrap_calibrations(self.cal, n_boot=17, seed=42, chunk_size=3)
        b = bootstrap_calibrations(self.cal, n_boot=17, seed=42, chunk_size=17)
        for key in ('cluster_counts', 'values', 'means', 'seconds'):
            np.testing.assert_allclose(a[key], b[key], rtol=1e-12, atol=1e-14)

    def test_heldout_scale_is_frozen(self):
        train = calibration_from_indices(self.cal, np.arange(100))
        test = calibration_from_indices(self.cal, np.arange(100, 200), train['coefficient'])
        np.testing.assert_allclose(test['times'], self.cal['work'][100:] * train['coefficient'])
        self.assertEqual(test['coefficient'], train['coefficient'])

    def test_all_proxies_and_paired_comparisons(self):
        for proxy in ('cost', 'output_tokens', 'total_tokens'):
            variant = load_calibration(proxy=proxy)
            self.assertAlmostEqual(variant['times'].mean(), 1.0, places=13)
            np.testing.assert_array_equal(variant['outcomes'], self.cal['outcomes'])
        boot = bootstrap_calibrations(self.cal, n_boot=20, seed=321)
        self.assertEqual(len(calibration_tables(self.cal, boot)['paired_gains']), 10)


if __name__ == '__main__':
    unittest.main()
