"""新版实验矩阵的分割、去 Spin、消融与恢复测试。"""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from core.experiment_setup import select_experiment_features, make_split_manifest, load_split_manifest
from core.train_optimizer import fit_preprocessing, prepare_cv_folds, optimize_model_parameters
from sklearn.linear_model import Lasso


class ExperimentChecks(unittest.TestCase):
    def test_feature_groups_and_spin_exclusion(self):
        features = pd.DataFrame({'MordredA': [1., 2.], 'Spin_1': [1, 1],
                                 'Morgan_0': [0, 1], 'MACCS_0': [0, 1], 'AtomPair_0': [0, 1],
                                 'TopoTorsion_0': [0, 1], 'RDKitFP_0': [0, 1]})
        self.assertEqual(list(select_experiment_features(features)), ['MordredA'])
        self.assertEqual(select_experiment_features(features, 'mixed').shape[1], 6)
        self.assertNotIn('Morgan_0', select_experiment_features(features, 'mixed', drop_fingerprint='Morgan'))
        self.assertEqual(list(select_experiment_features(features, 'mixed', only_fingerprint='MACCS')),
                         ['MordredA', 'MACCS_0'])
        self.assertNotIn('Spin_1', fit_preprocessing(features[['MordredA', 'Spin_1']], [1, 2])['feature_names'])

    def test_fixed_split_has_no_scaffold_leakage_and_ignores_target(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/'data.csv'
            data = pd.DataFrame({'A_number': np.repeat(np.arange(18), 10),
                                 'source_row': np.arange(180), 'SMILES': [f's{i}' for i in range(180)],
                                 'source_smiles': [f's{i}' for i in range(180)], 'Gap(eV)': np.arange(180)+1.,
                                 'Y_number': 1, 'C_number': 0})
            data.to_csv(source, index=False)
            manifest = make_split_manifest(data, source)
            changed = data.copy(); changed['Gap(eV)'] = changed['Gap(eV)'][::-1].to_numpy()
            second = make_split_manifest(changed, source)
            self.assertEqual(manifest['splits'], second['splits'])
            path = Path(folder)/'splits.json'; path.write_text(json.dumps(manifest), encoding='utf-8')
            tr, te, cv = load_split_manifest(path, data, source, 'scaffold')
            self.assertFalse(set(data.iloc[tr].A_number) & set(data.iloc[te].A_number))
            self.assertEqual(len(cv), 5)
            bad = dict(manifest)
            bad['splits']['scaffold']['test_positions'][0] = int(tr[0])
            path.write_text(json.dumps(bad), encoding='utf-8')
            with self.assertRaises(ValueError):
                load_split_manifest(path, data, source, 'scaffold')

    def test_optuna_resume_does_not_repeat_completed_trials(self):
        with tempfile.TemporaryDirectory() as folder:
            import sqlite3
            rng = np.random.RandomState(42)
            X = pd.DataFrame(rng.normal(size=(30, 4)), columns=list('abcd'))
            y = X['a'].to_numpy()*2 + rng.normal(size=30)*0.1
            folds = prepare_cv_folds(X,y)
            path=Path(folder)/'Lasso.sqlite3'
            optimize_model_parameters('Lasso', Lasso(random_state=42), folds, n_trials=1, study_path=path)
            optimize_model_parameters('Lasso', Lasso(random_state=42), folds, n_trials=1, study_path=path)
            with sqlite3.connect(path) as connection:
                self.assertEqual(connection.execute('SELECT COUNT(*) FROM trials').fetchone()[0], 1)
            connection.close()
            optimize_model_parameters('Lasso', Lasso(random_state=42), folds, n_trials=2, study_path=path)
            with sqlite3.connect(path) as connection:
                self.assertEqual(connection.execute('SELECT COUNT(*) FROM trials').fetchone()[0], 2)
            connection.close()


if __name__ == '__main__':
    unittest.main()
