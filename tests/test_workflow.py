"""针对数据对齐、CV 边界和模型配置的快速回归检查，不训练完整数据集。"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

from core.data_processing import standardize_smiles, sample_records, file_sha256, load_feature_cache
from core.train_optimizer import (
    target_correlations, fit_preprocessing, transform_features,
    prepare_cv_folds, create_models, shap_to_ev,
)


class WorkflowChecks(unittest.TestCase):
    def test_shuffled_index_keeps_target_pairs(self):
        X = pd.DataFrame({'a': [1., 2., 3., 4.]}, index=[9, 2, 18, 0])
        self.assertAlmostEqual(target_correlations(X, [2, 4, 6, 8])['a'], 1.0)

    def test_invalid_smiles_does_not_exit(self):
        for value in [None, np.nan, '', 'invalid smiles']:
            self.assertIsNone(standardize_smiles(value))
        self.assertEqual(standardize_smiles('C1=CC=CC=C1'), 'c1ccccc1')

    def test_validation_outlier_does_not_fit_preprocessing(self):
        X = pd.DataFrame({'a': [1., 2., np.nan, 4.], 'empty': [np.nan] * 4})
        prep = fit_preprocessing(X, [2, 3, 4, 6])
        self.assertEqual(prep['medians']['a'], 2.)
        self.assertNotIn('empty', prep['feature_names'])
        before = prep['feature_scaler'].mean_.copy()
        actual = transform_features(pd.DataFrame({'a': [100000., np.nan]}), prep)
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_array_equal(before, prep['feature_scaler'].mean_)

    def test_five_folds_learn_only_their_training_rows(self):
        rng = np.random.RandomState(8)
        X = pd.DataFrame(rng.normal(size=(30, 6)), columns=[f'f{i}' for i in range(6)], index=np.arange(30) * 3)
        y = np.arange(30, dtype=float) ** 2
        folds = prepare_cv_folds(X, y)
        self.assertEqual(len(folds), 5)
        seen = []
        for f in folds:
            tr, va = f['train_positions'], f['valid_positions']
            self.assertFalse(set(tr) & set(va))
            seen.extend(va)
            self.assertAlmostEqual(f['target_scaler'].mean_[0], y[tr].mean())
            names = f['preprocessing']['feature_names']
            np.testing.assert_allclose(f['preprocessing']['feature_scaler'].mean_, X.iloc[tr][names].mean())
        self.assertEqual(sorted(seen), list(range(30)))

    def test_validation_labels_do_not_change_fold_selection(self):
        rng = np.random.RandomState(11)
        X = pd.DataFrame(rng.normal(size=(30, 60)), columns=[f'f{i}' for i in range(60)])
        y = rng.normal(size=30)
        first = prepare_cv_folds(X, y)[0]
        changed_y = y.copy()
        changed_y[first['valid_positions']] = 100000.
        second = prepare_cv_folds(X, changed_y)[0]
        self.assertEqual(first['preprocessing']['feature_names'], second['preprocessing']['feature_names'])
        np.testing.assert_allclose(first['X_train'], second['X_train'])
        np.testing.assert_allclose(first['y_train'], second['y_train'])

    def test_group_folds_never_share_groups(self):
        X = pd.DataFrame({'a': np.arange(30, dtype=float)})
        groups = np.repeat(np.arange(10), 3)
        for f in prepare_cv_folds(X, np.arange(30), groups):
            self.assertFalse(set(groups[f['train_positions']]) & set(groups[f['valid_positions']]))

    def test_clone_preserves_gp_kernel_and_krr_is_rbf(self):
        models = create_models(smoke=True)
        original = models['GaussianProcess']
        candidate = clone(original).set_params(alpha=0.01)
        self.assertEqual(str(candidate.kernel), str(original.kernel))
        self.assertEqual(candidate.n_restarts_optimizer, original.n_restarts_optimizer)
        self.assertEqual(models['KernelRidge'].kernel, 'rbf')

    def test_shap_ev_additivity(self):
        prep = fit_preprocessing(pd.DataFrame({'a': [1., 2., 3.]}), [2., 4., 6.])
        values, base = shap_to_ev([[0.3, 0.7]], 0.5, prep['target_scaler'])
        predicted = prep['target_scaler'].inverse_transform([[1.5]])[0, 0]
        self.assertAlmostEqual(base + values.sum(), predicted)

    def test_cache_rejects_reorder_and_tamper(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            data = pd.DataFrame({'source_row': [0, 1], 'source_smiles': ['C', 'CC'],
                                 'Gap(eV)': [1., 2.], 'Y_number': [0, 0], 'C_number': [0, 1]})
            source, cache = path/'data.csv', path/'features.csv'
            data.to_csv(source, index=False)
            pd.DataFrame({'a': [1., 2.]}).to_csv(cache, index=False)
            with self.assertRaisesRegex(ValueError, '缺少'):
                load_feature_cache(cache, data, source)
            meta = {'schema_version': 1, 'raw_descriptors': True,
                    'data_sha256': file_sha256(source), 'cache_sha256': file_sha256(cache),
                    'samples': sample_records(data), 'columns': ['a']}
            Path(str(cache)+'.meta.json').write_text(json.dumps(meta), encoding='utf-8')
            self.assertEqual(load_feature_cache(cache, data, source)[0].shape, (2, 1))
            with self.assertRaisesRegex(ValueError, '校验失败'):
                load_feature_cache(cache, data.iloc[::-1], source)
            pd.DataFrame({'a': [2., 1.]}).to_csv(cache, index=False)
            with self.assertRaisesRegex(ValueError, '校验失败'):
                load_feature_cache(cache, data, source)


if __name__ == '__main__':
    unittest.main()
