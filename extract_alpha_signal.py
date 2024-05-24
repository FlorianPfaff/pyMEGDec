import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import scipy.signal

def extract_phase(signal, sampling_rate, lowcut=8.0, highcut=12.0):
    """
    Extracts the phase of the given signal using bandpass filtering and Hilbert transform.
    
    Parameters:
        signal (numpy array): The signal to extract the phase from.
        sampling_rate (float): The sampling rate of the signal.
        lowcut (float): The low cutoff frequency for the bandpass filter.
        highcut (float): The high cutoff frequency for the bandpass filter.
        
    Returns:
        numpy array: The phase of the filtered signal.
    """
    nyquist = 0.5 * sampling_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    order = 5  # Filter order
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    filtered_signal = scipy.signal.filtfilt(b, a, signal)
    analytic_signal = scipy.signal.hilbert(filtered_signal)
    phase = np.angle(analytic_signal)
    return phase

def average_phases(phases):
    """
    Averages the phases across multiple channels.
    
    Parameters:
        phases (list of numpy arrays): List of phase arrays from different channels.
        
    Returns:
        numpy array: The average phase.
    """
    phase_matrix = np.vstack(phases)
    mean_phase = np.angle(np.mean(np.exp(1j * phase_matrix), axis=0))
    return mean_phase

def extract_time_basis(data, trial_idx=0, channel_range=(187, 198)):
    """
    Extracts a robust time basis based on the alpha phases across multiple channels for a given trial.

    Parameters:
        data (dict): The filtered MEG data containing only the alpha signal.
        trial_idx (int): The index of the trial to extract the time basis from.
        channel_range (tuple): The range of channels (start, end) to extract phases for.
        
    Returns:
        numpy array: The robust time basis based on the average phase.
    """
    time_vector = data['time'][0][0][trial_idx][0]
    signal = data['trial'][0][0][trial_idx]
    
    sampling_rate = 1 / np.diff(time_vector[:2])[0]

    phases = []
    for channel_idx in range(channel_range[0], channel_range[1] + 1):
        signal_curr_chan = signal[channel_idx, :]
        phase = extract_phase(signal_curr_chan, sampling_rate)
        phases.append(phase)

    mean_phase = average_phases(phases)
    return mean_phase


if __name__ == "__main__":
    data_folder = r'.'
    part = 2
    data = sio.loadmat(f'{data_folder}/Part{part}Data.mat')['data'][0]

    # Extract the robust time basis for channels 187 to 198 for a specific trial
    time_basis = extract_time_basis(data, trial_idx=0, channel_range=(187, 198))

    # Display the time basis
    print("Robust time basis (average phase):", time_basis)

    # Plot the average phase to visualize
    time_vector = data['time'][0][0][0][0]
    plt.plot(time_vector, time_basis, label='Average Phase')
    plt.title('Average Alpha Phase Across Channels 187-198')
    plt.xlabel('Time (s)')
    plt.ylabel('Phase (radians)')
    plt.legend()
    plt.grid(True)
    plt.show()
