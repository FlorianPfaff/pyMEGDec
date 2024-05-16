import unittest
import numpy as np
from cross_validation import cross_validate_single_dataset

class TestCrossValidateSingleDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.data_folder = r'.'
        self.participant_id = 2
        self.n_folds = 10
        self.window_size = 0.1
        self.train_window_center = 0.2
        self.null_window_center = -0.2
        self.new_framerate = float('inf')
        self.classifier = 'multiclass-svm'
        self.classifier_param = np.nan
        self.components_pca = 200
        self.frequency_range = (0, float('inf'))

    def test_cross_validate_single_dataset_accuracy_svm(self):
        accuracy = cross_validate_single_dataset(self.data_folder, self.participant_id, self.n_folds, self.window_size, self.train_window_center, self.null_window_center, self.new_framerate, self.classifier, self.classifier_param, self.components_pca, self.frequency_range)
        
        self.assertGreaterEqual(accuracy, 0.25, "Accuracy should be at least 0.25")

    def test_cross_validate_single_dataset_accuracy_scikit_mlp(self):
        self.classifier = 'scikit-mlp'
        accuracy = cross_validate_single_dataset(self.data_folder, self.participant_id, self.n_folds, self.window_size, self.train_window_center, self.null_window_center, self.new_framerate, self.classifier, self.classifier_param, self.components_pca, self.frequency_range)
        
        self.assertGreaterEqual(accuracy, 0.15, "Accuracy should be at least 0.15")

if __name__ == '__main__':
    unittest.main()
