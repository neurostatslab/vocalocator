"""
Test the build model functions.
"""

import unittest

import torch
from constants import ENSEMBLE, ENSEMBLE_MISSING_COV, SIMPLENET_BASE, SIMPLENET_COV
from vocalocator.training.models import build_model


class TestOuptutShapes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # number of channels of input with four mics and passed-in cross correlations
        NUM_CHANNELS_MICS_AND_XCORR = 10
        cls.fake_input = torch.zeros(
            1, SIMPLENET_BASE["SAMPLE_LEN"], NUM_CHANNELS_MICS_AND_XCORR
        )

    def test_simplenet(self):
        # test loading the models
        cnn_no_cov, _ = build_model(SIMPLENET_BASE)
        cnn_cov, _ = build_model(SIMPLENET_COV)

        no_cov_expected_shape = (1, 2)
        cov_expected_shape = (1, 3, 2)

        self.assertEqual(cnn_no_cov(self.fake_input).shape, no_cov_expected_shape)
        self.assertEqual(cnn_cov(self.fake_input).shape, cov_expected_shape)

    def test_ensemble(self):
        ensemble, _ = build_model(ENSEMBLE)
        expected_shape = (1, len(ENSEMBLE["MODELS"]), 3, 2)
        self.assertEqual(ensemble(self.fake_input).shape, expected_shape)

        # test that the class throws an error if one of the models
        # doesn't have the 'OUTPUT_COV' flag
        with self.assertRaises(ValueError):
            build_model(ENSEMBLE_MISSING_COV)
