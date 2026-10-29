import copy
import csv
from pathlib import Path
import tempfile
import unittest
from strength_model import load_rows, fit_evaluate, screen, positive

DATA = 'data/synthetic_cube_results.csv'

class StrengthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = load_rows(DATA)
        cls.model = fit_evaluate(cls.rows)
    def test_batch_separation(self):
        self.assertFalse(set(self.model['train_batches']) & set(self.model['test_batches']))
        self.assertEqual(len(self.model['train_samples']),60)
        self.assertEqual(len(self.model['test_samples']),20)
    def test_all_rows_accounted_for(self):
        self.assertEqual(set(self.model['train_samples']+self.model['test_samples']), {r['sample_id'] for r in self.rows})
    def test_test_labels_do_not_affect_fit(self):
        changed=copy.deepcopy(self.rows)
        for row in changed:
            if row['batch_id'] in self.model['test_batches']: row['strength_28d_mpa']+=50
        other=fit_evaluate(changed)
        self.assertEqual(other['coefficient'],self.model['coefficient'])
        self.assertEqual(other['intercept'],self.model['intercept'])
        self.assertGreater(other['metrics']['mae_mpa'],40)
    def test_repeatable(self):
        self.assertEqual(self.model,fit_evaluate(self.rows))
    def test_hand_calculated_metrics(self):
        preds=self.model['held_out_predictions']
        expected=sum(abs(r['actual_28d_mpa']-r['predicted_28d_mpa']) for r in preds)/len(preds)
        self.assertAlmostEqual(expected,self.model['metrics']['mae_mpa'])
    def test_synthetic_baseline_comparison(self):
        self.assertLess(self.model['metrics']['mae_mpa'],self.model['metrics']['training_mean_baseline_mae_mpa'])
    def test_outside_range_abstains(self):
        self.assertIsNone(screen(self.model,100,40)['prediction_mpa'])
    def test_screening_boundary(self):
        simple={'training_range_7d_mpa':[10,30],'coefficient':1.5,'intercept':5}
        self.assertEqual(screen(simple,20,35)['status'],'at_or_above_screening_target')
        self.assertEqual(screen(simple,20,35.01)['status'],'below_screening_target')
    def test_nonfinite_and_negative(self):
        for n in [float('nan'),float('inf'),-1,0]:
            with self.assertRaises(ValueError): positive(n,'input')
    def test_duplicate_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bad.csv'
            content=Path(DATA).read_text()
            path.write_text(content+content.splitlines()[1]+'\n')
            with self.assertRaises(ValueError): load_rows(path)
    def test_too_few_batches(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bad.csv'; path.write_text('\n'.join(Path(DATA).read_text().splitlines()[:5]))
            with self.assertRaises(ValueError): load_rows(path)

if __name__=='__main__': unittest.main()
