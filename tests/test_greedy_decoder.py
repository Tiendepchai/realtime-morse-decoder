import unittest
import torch
from src.utils.text import greedy_decoder, TextTransform, CHAR2IDX

class TestGreedyDecoder(unittest.TestCase):
    def setUp(self):
        self.transform = TextTransform()
        self.vocab_size = 38 # 37 chars + 0 blank

    def test_basic_decoding(self):
        # T=5, B=1, C=38
        # Sequence: Blank, A, Blank, B, Blank -> "AB"
        T, B, C = 5, 1, self.vocab_size
        log_probs = torch.zeros(T, B, C) - 100.0 # mostly small
        
        # Set max probs
        # t=0: Blank(0)
        log_probs[0, 0, 0] = 0.0
        # t=1: A
        log_probs[1, 0, CHAR2IDX["A"]] = 0.0
        # t=2: Blank
        log_probs[2, 0, 0] = 0.0
        # t=3: B
        log_probs[3, 0, CHAR2IDX["B"]] = 0.0
        # t=4: Blank
        log_probs[4, 0, 0] = 0.0
        
        output_lengths = torch.tensor([5])
        labels = torch.tensor([CHAR2IDX["A"], CHAR2IDX["B"]])
        label_lengths = torch.tensor([2])
        
        decoded, targets = greedy_decoder(log_probs, output_lengths, labels, label_lengths, self.transform)
        
        self.assertEqual(decoded[0], "AB")
        self.assertEqual(targets[0], "AB")

    def test_repeating_characters(self):
        # CTC rule: A A -> A, A Blank A -> AA
        # Sequence: A, A, Blank, A -> "AA"
        T, B, C = 4, 1, self.vocab_size
        log_probs = torch.zeros(T, B, C) - 100.0
        
        log_probs[0, 0, CHAR2IDX["A"]] = 0.0
        log_probs[1, 0, CHAR2IDX["A"]] = 0.0 # Check collapse
        log_probs[2, 0, 0] = 0.0
        log_probs[3, 0, CHAR2IDX["A"]] = 0.0
        
        output_lengths = torch.tensor([4])
        labels = torch.tensor([CHAR2IDX["A"], CHAR2IDX["A"]])
        label_lengths = torch.tensor([2])
        
        decoded, targets = greedy_decoder(log_probs, output_lengths, labels, label_lengths, self.transform)
        
        self.assertEqual(decoded[0], "AA")

    def test_batch_processing(self):
        # Batch of 2
        # b1: "A" (len 3) -> 0 A 0
        # b2: "B" (len 2) -> B 0 ...
        T, B, C = 3, 2, self.vocab_size
        log_probs = torch.zeros(T, B, C) - 100.0
        
        # Batch 0
        log_probs[0, 0, 0] = 0.0
        log_probs[1, 0, CHAR2IDX["A"]] = 0.0
        log_probs[2, 0, 0] = 0.0
        
        # Batch 1
        log_probs[0, 1, CHAR2IDX["B"]] = 0.0
        log_probs[1, 1, 0] = 0.0
        # t=2 ignored for b1 if len=2
        
        output_lengths = torch.tensor([3, 2])
        labels = torch.tensor([CHAR2IDX["A"], CHAR2IDX["B"]])
        label_lengths = torch.tensor([1, 1])
        
        decoded, targets = greedy_decoder(log_probs, output_lengths, labels, label_lengths, self.transform)
        
        self.assertEqual(decoded[0], "A")
        self.assertEqual(decoded[1], "B")

if __name__ == "__main__":
    unittest.main()
