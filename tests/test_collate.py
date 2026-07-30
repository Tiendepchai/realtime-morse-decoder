import unittest
import torch
import torch.nn as nn
from src.train import collate_fn

class TestCollate(unittest.TestCase):
    def test_collate_fn_basic(self):
        # Batch of 2
        # Item 1: Feat (100, 64), Label (5,)
        # Item 2: Feat (50, 64), Label (3,)
        
        F = 64
        feat1 = torch.randn(100, F)
        label1 = torch.tensor([1, 2, 3, 4, 5])
        
        feat2 = torch.randn(50, F)
        label2 = torch.tensor([6, 7, 8])
        
        batch = [(feat1, label1), (feat2, label2)]
        
        features_padded, labels_concatenated, input_lengths, label_lengths = collate_fn(batch)
        
        # Check Shapes
        # features_padded: (B, 1, F, T) -> (2, 1, 64, 100)
        self.assertEqual(features_padded.shape, (2, 1, 64, 100))
        
        # input_lengths: (100, 50)
        self.assertTrue(torch.equal(input_lengths, torch.tensor([100, 50])))
        
        # checks padding (should be 0)
        # item 2, from time 50 to 99 should be 0
        self.assertTrue(torch.all(features_padded[1, 0, :, 50:] == 0))
        
        # labels_concatenated: (8,)
        self.assertEqual(labels_concatenated.shape, (8,))
        self.assertTrue(torch.equal(labels_concatenated, torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])))
        
        # label_lengths: (5, 3)
        self.assertTrue(torch.equal(label_lengths, torch.tensor([5, 3])))

if __name__ == "__main__":
    unittest.main()
