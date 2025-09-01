import unittest
import numpy as np
from evaluation.metrics import compute_accuracy, compute_per_class_accuracy

class TestMetrics(unittest.TestCase):
    def test_accuracy_calculation(self):
        labels = np.array([0, 1, 2, 0])
        # Perfect predictions
        preds = np.zeros((4, 3))
        preds[0, 0] = 1.0
        preds[1, 1] = 1.0
        preds[2, 2] = 1.0
        preds[3, 0] = 1.0
        
        acc = compute_accuracy(preds, labels)
        self.assertEqual(acc, 1.0)
        
        # 50% accuracy
        preds[3, 1] = 2.0 # wrong
        acc = compute_accuracy(preds, labels)
        self.assertEqual(acc, 0.75)

    def test_per_class_accuracy(self):
        classes = ["a", "b"]
        labels = [0, 0, 1, 1]
        preds = np.array([
            [1.0, 0.0], # correct
            [0.0, 1.0], # wrong
            [0.0, 1.0], # correct
            [0.0, 1.0]  # correct
        ])
        
        stats = compute_per_class_accuracy(preds, labels, classes)
        self.assertEqual(stats["a"], "50.00%")
        self.assertEqual(stats["b"], "100.00%")

if __name__ == '__main__':
    unittest.main()
