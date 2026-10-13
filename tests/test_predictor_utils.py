"""Small offline tests for label formatting and care guidance."""
import unittest

from src.predictor import care_guidance, format_label


class PredictorUtilityTests(unittest.TestCase):
    def test_format_label_with_crop_and_disease(self):
        crop, disease, display = format_label("Tomato___Early_blight")
        self.assertEqual(crop, "Tomato")
        self.assertEqual(disease, "Early blight")
        self.assertEqual(display, "Tomato — Early blight")

    def test_format_label_without_separator(self):
        crop, disease, display = format_label("unknown_class")
        self.assertEqual(crop, "Unknown crop")
        self.assertEqual(disease, "unknown class")
        self.assertIn("unknown class", display)

    def test_healthy_guidance(self):
        intro, advice = care_guidance("healthy")
        self.assertIn("healthy", intro.lower())
        self.assertGreaterEqual(len(advice), 2)

    def test_disease_guidance_is_cautious(self):
        intro, advice = care_guidance("Early blight")
        self.assertIn("visual pattern", intro)
        self.assertTrue(any("agricultural" in item.lower() for item in advice))


if __name__ == "__main__":
    unittest.main()
