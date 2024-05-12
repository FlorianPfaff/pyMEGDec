import unittest
import numpy as np
from evaluate_model_transfer import evaluate_model_transfer

class TestEvaluateModelTransfer(unittest.TestCase):
    def test_evaluate_model_transfer_accuracy(self):
        data_folder = r'.'
        parts = 2
        null_window_center = np.nan
        classifier = 'multiclass-svm'
        
        accuracy = evaluate_model_transfer(data_folder, parts, null_window_center=null_window_center, classifier=classifier)
        
        self.assertGreaterEqual(accuracy, 0.25, "Accuracy should be at least 0.25")

if __name__ == '__main__':
    unittest.main()
