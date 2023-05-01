import os
import unittest

import pandas as pd

from model import batch_predict


class TestBatchPredict(unittest.TestCase):
    def setUp(self):
        # Define some test data
        self.test_df = pd.DataFrame(
            {
                "Type": ["Cat", "Cat"],
                "Age": [3, 12],
                "Breed1": ["Tabby", "Domestic Medium Hair"],
                "Gender": ["Male", "Female"],
                "Color1": ["Black", "Black"],
                "Color2": ["White", "White"],
                "MaturitySize": ["Small", "Medium"],
                "FurLength": ["Short", "Medium"],
                "Vaccinated": ["No", "Not Sure"],
                "Sterilized": ["No", "Not Sure"],
                "Health": ["Healthy", "Healthy"],
                "Fee": [100, 0],
                "PhotoAmt": [1, 2],
                "target": ["Yes", "No"],
            }
        )

        # Define some test configuration data
        self.model_data_cfg = {"target_dir": "artifacts/", "model_file": "model.joblib"}
        self.test_data_cfg = {"target_dir": "output", "output_file": "test-results.csv"}

    def test_batch_predict(self):
        # Test the batch_predict function
        batch_predict(self.test_df, "target", self.model_data_cfg, self.test_data_cfg)

        # Assert that the results file was created
        result_file = os.path.join(
            self.test_data_cfg["target_dir"], self.test_data_cfg["output_file"]
        )
        self.assertTrue(os.path.exists(result_file))

        # Assert that the output file contains the expected columns
        expected_cols = [x if x != "target" else "Adopted" for x in self.test_df.columns] + [
            "Adopted_prediction"
        ]
        result_df = pd.read_csv(result_file)
        self.assertListEqual(list(result_df.columns), expected_cols)

        # Assert that the output file contains the expected number of rows
        self.assertEqual(result_df.shape[0], self.test_df.shape[0])


if __name__ == "__main__":
    unittest.main()
