import numpy as np
import scipy.io as sio
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Reuse functions from the previous script
from evaluate_model_transfer import (preprocess_features, reduce_features_pca, 
                             train_multiclass_classifier, get_default_classifier_param)


def cross_validate_single_dataset(data_folder, participant_id, n_folds=10, window_size=0.1, train_window_center=0.2,
                                  null_window_center=-0.2, new_framerate=float('inf'), classifier='multiclass-svm', 
                                  classifier_param=np.nan, components_pca=100, frequency_range=(0, float('inf'))):

    if np.isnan(classifier_param):
        classifier_param = get_default_classifier_param(classifier)

    data = sio.loadmat(f'{data_folder}/Part{participant_id}Data.mat')['data'][0]
    labels = data['trialinfo'][0][0]
    n_trials = len(labels)
    n_stim = len(np.unique(labels))
    pred_lbl = np.empty(n_trials)
    pred_lbl.fill(np.nan)

    stimuli_features, null_features = preprocess_features(data, frequency_range, new_framerate, window_size, train_window_center, null_window_center)
    all_features = np.vstack((stimuli_features, null_features)).squeeze().T

    fold = np.ceil(np.arange(1, data['trial'][0].shape[1] + 1) / (data['trial'][0].shape[1] / n_folds)).astype(int)
    if null_features:
        fold_aug = np.concatenate((fold, fold))
        labels = np.concatenate((labels, np.zeros(n_trials, dtype=int)))

    n_trials_per_fold = n_trials // n_folds

    for f in range(1, n_folds + 1):
        train_features = all_features[:, fold_aug != f].T
        train_labels = labels[fold_aug != f]
        test_features = all_features[:, (fold_aug == f) & (labels != 0)].T

        if components_pca != float('inf'):
            train_features, coeff, train_feature_mean, explained_variance = reduce_features_pca(train_features, components_pca)
            print(f'Explained Variance by {components_pca} components: {explained_variance:.2f}%')
            test_features = (test_features - train_feature_mean) @ coeff[:, :components_pca]

        if classifier in ['gradient-boosting', 'lasso', 'svm-binary']:
            all_pred = np.zeros((n_trials_per_fold, n_stim))
            for stim in range(1, n_stim + 1):
                if classifier == 'gradient-boosting':
                    model = train_gradient_boosting(train_features, train_labels == stim, classifier_param)
                elif classifier == 'lasso':
                    model = train_for_stimulus_lasso_glm(train_features, train_labels == stim, classifier_param)
                elif classifier == 'svm-binary':
                    model = train_binary_svm(train_features, train_labels == stim, classifier_param)
                all_pred[:, stim - 1] = model.predict_proba(test_features)[:, 1] if classifier == 'svm-binary' else model.predict(test_features)
            pred_lbl[fold == f] = np.argmax(all_pred, axis=1) + 1
        else:
            model = train_multiclass_classifier(train_features, train_labels, classifier, classifier_param)
            pred_lbl[fold == f] = model.predict(test_features)

    if np.all(pred_lbl == 0):
        print("All predictions are the null-class. Replace them to be fair with the binary classifiers for which one always decides on a label unequal to 0. Using 1.")
        pred_lbl[pred_lbl == 0] = 1
    elif np.any(pred_lbl == 0):
        print("Some predictions are the null-class. Replace them to be fair with the binary classifiers for which one always decides on a label unequal to 0. Using least frequent label.")
        nonzero_labels, counts = np.unique(pred_lbl[pred_lbl > 0], return_counts=True)
        min_label = nonzero_labels[np.argmin(counts)]
        pred_lbl[pred_lbl == 0] = min_label

    accuracy = np.mean(labels[labels > 0] == pred_lbl)
    print(f'Participant {participant_id}: {accuracy * 100:.2f}% accuracy')

    return accuracy


def train_gradient_boosting(train_features, train_labels, classifier_param):
    model = GradientBoostingClassifier(n_estimators=int(classifier_param), max_depth=3, learning_rate=0.1)
    model.fit(train_features, train_labels)
    return model


def train_for_stimulus_lasso_glm(train_features, train_labels, lambda_):
    model = LogisticRegression(penalty='l1', C=lambda_, solver='liblinear')
    model.fit(train_features, train_labels)
    return model


def train_binary_svm(train_features, train_labels, box_constraint):
    model = SVC(C=box_constraint, kernel='linear', probability=True)
    model.fit(train_features, train_labels)
    return model


if __name__ == '__main__':
    acc = cross_validate_single_dataset(r'.', 2, classifier='multiclass-svm', components_pca=100)
    print(acc)
