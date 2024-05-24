import numpy as np
import re
import matplotlib.pyplot as plt
import scipy.io as sio
from extract_alpha_signal import extract_phase

def extract_channels_by_location(data, location_pattern):
    """
    Extracts channel indices by location from the data.

    Parameters:
        data (dict): The filtered MEG data containing only the alpha signal.
        location_pattern (str): The regex pattern for the desired location.

    Returns:
        list: List of indices of the channels matching the location pattern.
    """
    channel_names = [subarray[0][0] for subarray in data[0][0]]
    extracted_strings = [(index, string) for index, string in enumerate(channel_names)]

    pattern = re.compile(location_pattern)
    channel_indices = [index for index, string in extracted_strings if pattern.match(string)]
    
    return channel_indices

def plot_all_alpha_signals(time_vector, signals, channel_indices, time_window):
    """
    Plots the alpha signals for all specified channels in a single plot.

    Parameters:
        time_vector (np.ndarray): The time vector.
        signals (list of np.ndarray): The alpha signals for all channels.
        channel_indices (list): List of indices of the specified channels.
        time_window (tuple): The time window (start, end) to visualize in seconds.
    """
    plt.figure()
    for idx, signal in zip(channel_indices, signals):
        plt.plot(time_vector, signal, label=f'Channel {idx}')
    
    plt.title('Alpha Signals (8-12 Hz) - Selected Channels')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.xlim(time_window)
    plt.legend()
    plt.tight_layout()
    plt.show()

def calculate_phase_differences(phases):
    """
    Calculates phase differences between channels.

    Parameters:
        phases (list): List of phases for each channel.

    Returns:
        np.ndarray: Matrix of phase differences between channels.
    """
    num_channels = len(phases)
    phase_diffs = np.zeros((num_channels, num_channels))

    for i in range(num_channels):
        for j in range(num_channels):
            phase_diffs[i, j] = np.mean(np.abs(phases[i] - phases[j]))
    
    return phase_diffs

def plot_phase_differences(phase_diffs, channel_indices):
    """
    Plots a confusion matrix-like plot for phase differences between channels.

    Parameters:
        phase_diffs (np.ndarray): Matrix of phase differences between channels.
        channel_indices (list): List of indices of the specified channels.
    """
    num_channels = len(channel_indices)
    plt.figure(figsize=(8, 6))
    plt.imshow(phase_diffs, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Phase Difference (radians)')
    plt.title('Phase Differences Between Selected Channels')
    plt.xlabel('Channel Index')
    plt.ylabel('Channel Index')
    plt.xticks(ticks=np.arange(num_channels), labels=channel_indices, rotation=90)
    plt.yticks(ticks=np.arange(num_channels), labels=channel_indices)
    plt.show()

def extract_phases_and_channels(data, trial_idx, location_pattern):
    """
    Extracts the phases and channel indices for a given trial and location pattern.

    Parameters:
        data (dict): The filtered MEG data containing only the alpha signal.
        trial_idx (int): The index of the trial to visualize.
        location_pattern (str): The regex pattern for the desired location channels.

    Returns:
        tuple: Phases and channel indices.
    """
    time_vector = data['time'][0][0][trial_idx][0]
    signal = data['trial'][0][0][trial_idx]
    
    sampling_rate = 1 / np.diff(time_vector[:2])[0]
    channel_indices = extract_channels_by_location(data, location_pattern)

    phases = []
    signals = []
    for channel_idx in channel_indices:
        signal_curr_chan = signal[channel_idx, :]
        phase = extract_phase(signal_curr_chan, sampling_rate)
        phases.append(phase)
        signals.append(signal_curr_chan)
    
    return phases, channel_indices, time_vector, signals

def show_bandpass_filtered_signals(data, trial_idx=0, time_window=(0, 1), location_pattern=r'^M.O..$'):
    """
    Visualizes the alpha signal for a specific trial and extracts the phase.

    Parameters:
        data (dict): The filtered MEG data containing only the alpha signal.
        trial_idx (int): The index of the trial to visualize.
        time_window (tuple): The time window (start, end) to visualize in seconds.
        location_pattern (str): The regex pattern for the desired location channels.
    """
    phases, channel_indices, time_vector, signals = extract_phases_and_channels(data, trial_idx, location_pattern)
    plot_all_alpha_signals(time_vector, signals, channel_indices, time_window)
    return phases, channel_indices

def visualize_phase_shifts(data, trial_idx=0, location_pattern=r'^M.O..$'):
    """
    Visualizes phase shifts between specified channels.

    Parameters:
        data (dict): The filtered MEG data containing only the alpha signal.
        trial_idx (int): The index of the trial to visualize.
        location_pattern (str): The regex pattern for the desired location channels.
    """
    phases, channel_indices, _, _ = extract_phases_and_channels(data, trial_idx, location_pattern)
    phase_diffs = calculate_phase_differences(phases)
    plot_phase_differences(phase_diffs, channel_indices)

if __name__ == "__main__":
    trial_idx = 0
    time_window = (0, 1)
    data_folder = r'C:\Users\emper\owncloud-ld\Persönlich\Neuroscience'
    part = 2
    location_pattern = r'^M.O..$'

    data = sio.loadmat(f'{data_folder}/Part{part}Data.mat')['data'][0]
    show_bandpass_filtered_signals(data, trial_idx, time_window, location_pattern)
    visualize_phase_shifts(data, trial_idx, location_pattern)