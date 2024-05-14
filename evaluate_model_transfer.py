import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch import optim
import torch.nn as nn
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
import torch
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.neural_network import MLPClassifier
import warnings
import xgboost as xgb

def evaluate_model_transfer(data_folder, parts, window_size=0.1, train_window_center=0.2, 
                            null_window_center=-0.2, new_framerate=float('inf'), classifier='multiclass-svm', 
                            classifier_param=np.nan, components_pca=100, frequency_range=(0, float('inf'))):

    if not isinstance(classifier_param, dict) and np.all(np.isnan(classifier_param)):
        classifier_param = get_default_classifier_param(classifier)
    
    train_exp_data = sio.loadmat(f'{data_folder}/Part{parts}Data.mat')['data'][0]
    val_exp_data = sio.loadmat(f'{data_folder}/Part{parts}CueData.mat')['data'][0]
    
    labels_train_exp = train_exp_data['trialinfo'][0][0]
    labels_val_exp = val_exp_data['trialinfo'][0][0]
    if np.isnan(null_window_center):
        """ Subtract 1 from the labels because there is no null data in the validation experiment 
        and some classifiers do not labels starting from 0."""
        labels_train_exp -= 1
        labels_val_exp -= 1

    assert np.allclose(np.diff(val_exp_data['time'][0][0][0][0, :2]), np.diff(val_exp_data['time'][0][0][0][0, :2])), \
        'Sampling rate of the two experiments must match.'
    
    if not np.array_equal(np.unique(labels_train_exp), np.unique(labels_val_exp)):
        warnings.warn('There are labels in the training or validation experiment that are not in the other experiment.')
    
    stimuli_features_train_exp, null_features_train_exp = preprocess_features(
        train_exp_data, frequency_range, new_framerate, window_size, train_window_center, null_window_center)
    stimuli_features_val_exp, _ = preprocess_features(
        val_exp_data, frequency_range, new_framerate, window_size, train_window_center, np.nan)

    features_train_exp = np.hstack(stimuli_features_train_exp + null_features_train_exp).T
    labels_train_exp = np.concatenate((labels_train_exp, np.zeros(len(null_features_train_exp), dtype=int)))

    features_val_exp = np.hstack(stimuli_features_val_exp).T
    
    if components_pca != float('inf'):
        features_train_exp, coeff, features_train_exp_mean, explained_variance = reduce_features_pca(features_train_exp, components_pca)
        print(f'Explained Variance by {components_pca} components: {explained_variance:.2f}%')
        features_val_exp = (features_val_exp - features_train_exp_mean) @ coeff[:, :components_pca]
    
    model = train_multiclass_classifier(features_train_exp, labels_train_exp, classifier, classifier_param)
    predictions_val_exp = model.predict(features_val_exp)

    accuracy = np.mean(predictions_val_exp == labels_val_exp)
    return accuracy

def preprocess_features(data, frequency_range, new_framerate, window_size, train_window_center, null_window_center):
    data = filter_features(data, frequency_range[0], frequency_range[1])
    if new_framerate != float('inf'):
        data = downsample_data(data, new_framerate)
    train_window = (train_window_center - window_size / 2, train_window_center + window_size / 2)
    null_time_window = (null_window_center - window_size / 2, null_window_center + window_size / 2) if not np.isnan(null_window_center) else (np.nan, np.nan)
    assert np.isnan(null_time_window).all() or null_time_window[1] <= train_window[0], 'Null window must be before train window'
    stimuli_features_cell, null_features_cell = extract_windows(data, train_window, null_time_window)
    return stimuli_features_cell, null_features_cell


def filter_features(data, low_freq, high_freq):
    if not data['time'][0][0][0].size:
        raise ValueError('Time vector is empty or not provided correctly.')

    Fs = 1 / np.diff(data['time'][0][0][0][0,:2])
    
    assert low_freq >= 0, 'Low frequency must be greater than or equal to 0'
    assert high_freq >= 0, 'High frequency must be greater than or equal to 0'
    assert high_freq >= low_freq, 'High frequency must be greater than or equal to low frequency'
    
    if low_freq == 0 and high_freq == float('inf'):
        return data
    elif low_freq == 0:
        b, a = butter(4, high_freq / (Fs / 2), 'low')
    elif high_freq != float('inf'):
        b, a = butter(4, [low_freq, high_freq] / (Fs / 2), 'bandpass')
    else:
        raise ValueError("Highpass filter not supported.")

    for i in range(len(data['trial'][0])):
        data['trial'][0][i] = filtfilt(b, a, data['trial'][0][i].T).T
    return data


def downsample_data(data, new_framerate):
    raw_fs = 1 / np.diff(data['time'][0][0][0][0,:2])
    if new_framerate != raw_fs:
        new_t = np.arange(data['time'][0][0][0][0,0], data['time'][0][0][0][0,-1], 1 / new_framerate)
        for i in range(len(data['trial'][0][0])):
            interpolator = interp1d(data['time'][0][0][i][0,:], data['trial'][0][0][i], axis=1, fill_value="extrapolate")
            data['trial'][0][0][i] = interpolator(new_t)
            data['time'][0][0][i] = new_t[None]
    return data


def extract_windows(data, train_window, null_time_window):
    train_begin_index = np.argmin(np.abs(data['time'][0][0][0] - train_window[0]))
    train_end_index = np.argmin(np.abs(data['time'][0][0][0] - train_window[1]))
    stimuli_features_cell = [trial[:, train_begin_index:train_end_index].reshape(-1, 1) for trial in data['trial'][0][0]]

    if np.isnan(null_time_window).all():
        null_features_cell = []
    elif null_time_window[1]-null_time_window[0] >= 0:
        null_begin_index = np.argmin(np.abs(data['time'][0][0][0] - null_time_window[0]))
        null_end_index = null_begin_index + (train_end_index - train_begin_index)
        null_features_cell = [trial[:, null_begin_index:null_end_index].reshape(-1, 1) for trial in data['trial'][0][0]]
    else:
        raise ValueError('Invalid null window')

    return stimuli_features_cell, null_features_cell


def reduce_features_pca(features, n_components):
    pca = PCA(n_components=n_components)
    features_train_exp_mean = np.mean(features, axis=0)
    features_centered = features - features_train_exp_mean
    reduced_features = pca.fit_transform(features_centered)
    explained_variance = np.sum(pca.explained_variance_ratio_) * 100
    return reduced_features, pca.components_.T, features_train_exp_mean, explained_variance


def train_multiclass_classifier(features, labels, classifier, classifier_param):
    if classifier == 'multiclass-svm':
        model = SVC(C=classifier_param, probability=True)
    elif classifier == 'random-forest':
        model = RandomForestClassifier(n_estimators=int(classifier_param))
    elif classifier == 'gradient-boosting':
        model = GradientBoostingClassifier(n_estimators=int(classifier_param))
    elif classifier == 'knn':
        model = KNeighborsClassifier(n_neighbors=int(classifier_param))
    elif classifier == 'mostFrequentDummy':
        model = DummyClassifier(strategy='most_frequent')
    elif classifier == 'always1Dummy':
        model = DummyClassifier(strategy='constant', constant=1)
    elif classifier == 'xgboost':
        model = xgb.XGBClassifier(n_estimators=int(classifier_param), use_label_encoder=False, eval_metric='mlogloss')
    elif classifier == 'scikit-mlp':
        model = MLPClassifier(hidden_layer_sizes=int(classifier_param[0]), max_iter=int(classifier_param[1]))
    elif classifier == 'pytorch-mlp':
        # Define model dimensions
        input_dim = features.shape[1]
        output_dim = len(np.unique(labels))

        # Instantiate the model
        model = MLPClassifierTorch(input_dim, int(classifier_param["hidden_dim"]),
                                   output_dim, learning_rate=classifier_param["learning_rate"],
                                   dropout_rate=classifier_param["dropout_rate"])

        # Create the full dataset
        full_dataset = TensorDataset(torch.tensor(features, dtype=torch.float32), 
                                    torch.tensor(labels, dtype=torch.long))

        # Define the size of training and validation datasets
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size

        # Split the dataset into training and validation sets
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        # Create data loaders for training and validation sets
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

        # Configure the trainer to use the GPU

        trainer = pl.Trainer(
            max_epochs=int(classifier_param["max_epochs"]),
            default_root_dir=r"C:\Users\emper\lightning_logs",
            callbacks=[pl.callbacks.EarlyStopping(monitor='val_loss', patience=10)]
        )
       
        # Train the model
        trainer.fit(model, train_loader, val_loader)
        return model
    else:
        raise ValueError(f"Unsupported classifier: {classifier}")
    
    model.fit(features, labels)
    return model


def get_default_classifier_param(classifier):
    if classifier == 'multiclass-svm':
        return 3.0  # Default C value
    elif classifier == 'random-forest':
        return 250  # Default number of trees
    elif classifier == 'gradient-boosting':
        return 100  # Default number of boosting iterations
    elif classifier == 'knn':
        return 5  # Default number of neighbors
    elif classifier in ['mostFrequentDummy', 'always1Dummy']:
        return None
    elif classifier == 'xgboost':
        return 100  # Default number of boosting iterations
    elif classifier == 'scikit-mlp':
        return (150, 1000)
    elif classifier == 'pytorch-mlp':
        return {'hidden_dim': 720, 'max_epochs': 500, 'learning_rate': 1e-3, 'dropout_rate': 0.2}
    else:
        raise ValueError(f"Unsupported classifier: {classifier}")


class MLPClassifierTorch(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=1e-3, dropout_rate=0.2):
        super(MLPClassifierTorch, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, output_dim)
        self.learning_rate = learning_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()       
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.dropout(x)
        x = self.relu(self.layer_2(x))
        x = self.dropout(x)
        x = self.layer_3(x)
        return x
    
    def predict(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient computation
            predictions = self.relu(self.layer_1(x))
            predictions = self.relu(self.layer_2(predictions))
            predictions = self.layer_3(predictions)
        return torch.argmax(predictions, dim=1).numpy()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)
        return optimizer


if __name__ == '__main__':
    acc = evaluate_model_transfer(r'.', 2, classifier='multiclass-svm', components_pca=100)
    print(acc)