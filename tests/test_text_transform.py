import unittest
from src.utils.text import TextTransform, CHARS, CHAR2IDX

class TestTextTransform(unittest.TestCase):
    def setUp(self):
        self.transform = TextTransform()

    def test_text_to_int_basic(self):
        text = "ABC"
        expected = [CHAR2IDX["A"], CHAR2IDX["B"], CHAR2IDX["C"]]
        result = self.transform.text_to_int(text)
        self.assertEqual(result, expected)

    def test_text_to_int_lowercase(self):
        text = "abc"
        expected = [CHAR2IDX["A"], CHAR2IDX["B"], CHAR2IDX["C"]]
        result = self.transform.text_to_int(text)
        self.assertEqual(result, expected)

    def test_text_to_int_with_spaces(self):
        text = "SOS HELP"
        expected = [CHAR2IDX["S"], CHAR2IDX["O"], CHAR2IDX["S"], CHAR2IDX[" "],
                    CHAR2IDX["H"], CHAR2IDX["E"], CHAR2IDX["L"], CHAR2IDX["P"]]
        result = self.transform.text_to_int(text)
        self.assertEqual(result, expected)

    def test_text_to_int_invalid_chars(self):
        text = "A.B!C" # ., ! are not in CHARS
        expected = [CHAR2IDX["A"], CHAR2IDX["B"], CHAR2IDX["C"]]
        result = self.transform.text_to_int(text)
        self.assertEqual(result, expected)

    def test_int_to_text(self):
        seq = [CHAR2IDX["A"], CHAR2IDX["B"], CHAR2IDX[" "], CHAR2IDX["C"]]
        expected = "AB C"
        result = self.transform.int_to_text(seq)
        self.assertEqual(result, expected)

    def test_roundtrip(self):
        text = "THE QUICK BROWN FOX 0123456789"
        integers = self.transform.text_to_int(text)
        decoded = self.transform.int_to_text(integers)
        self.assertEqual(decoded, text)

if __name__ == "__main__":
    unittest.main()
