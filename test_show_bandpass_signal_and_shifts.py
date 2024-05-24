import unittest
from unittest.mock import patch
import scipy.io as sio
from show_bandpass_signal_and_shifts import (
    extract_channels_by_location,
    show_bandpass_filtered_signals,
    visualize_phase_shifts
)

class TestAlphaChannelMethods(unittest.TestCase):
    
    def setUp(self):
        self.data_folder = r'.'
        self.part = 2
        self.data = sio.loadmat(f'{self.data_folder}/Part{self.part}Data.mat')['data'][0]

    def test_extract_channels_by_location(self):
        occipital_pattern = r'^M.O..$'
        occipital_indices = extract_channels_by_location(self.data, occipital_pattern)
        self.assertGreater(len(occipital_indices), 10)
    
    @patch('show_bandpass_signal_and_shifts.plt.show')
    def test_show_bandpass_filtered_signals(self, mock_show):
        trial_idx = 0
        time_window = (0, 1)
        location_pattern = r'^M.O..$'
        
        phases, _ = show_bandpass_filtered_signals(self.data, trial_idx, time_window, location_pattern)
        self.assertGreater(len(phases), 10)
        mock_show.assert_called_once()
    
    @patch('show_bandpass_signal_and_shifts.plt.show')
    def test_visualize_phase_shifts(self, mock_show):
        trial_idx = 0
        location_pattern = r'^M.O..$'
        
        visualize_phase_shifts(self.data, trial_idx, location_pattern)
        # Assuming visualize_phase_shifts does not return a value but generates a plot.
        self.assertIsNone(visualize_phase_shifts(self.data, trial_idx, location_pattern))
        mock_show.assert_called()

if __name__ == "__main__":
    unittest.main()
