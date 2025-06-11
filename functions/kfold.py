import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import RobustScaler

def get_fold_boundaries(data_input, cv_n, random_state):
    data_input = data_input.sample(frac = 1, random_state = random_state, ignore_index = True)
    partition_size = len(data_input) // cv_n
    fold_boundaries = np.arange(0, len(data_input), partition_size)
    return [(fold_boundaries[i - 1].item(), fold_boundaries[i].item() - 1) for i in range(1, len(fold_boundaries))]

def get_tdata(data_input, fold_boundaries, fold_index):
    t0_tuples = fold_boundaries[0:fold_index]
    t1_tuples = fold_boundaries[fold_index + 1:]

    if len(t0_tuples) != 0:
            t0_start = t0_tuples[0][0]
            t0_end = t0_tuples[-1][-1]

            t0_data = data_input.iloc[t0_start:t0_end + 1]
    else:
        t0_data = pd.DataFrame()

    if len(t1_tuples) != 0:
            t1_start = t1_tuples[0][0]
            t1_end = t1_tuples[-1][-1]
        
            t1_data = data_input.iloc[t1_start:t1_end + 1]
    else:
        t1_data = pd.DataFrame()

    t_data = pd.concat([t0_data, t1_data], axis = 0, ignore_index = True)

    return t_data

def scale_data(t_data, v_data, scaler, sel_cols):
    nparray_scaled_features = scaler.fit_transform(t_data[sel_cols])
    t_data.loc[:, sel_cols] = nparray_scaled_features

    nparray_scaled_features = scaler.transform(v_data[sel_cols])
    v_data.loc[:, sel_cols] = nparray_scaled_features

    return t_data, v_data

def data_balancing(t_data, balancing_method):
    if balancing_method == "over":
        oversample = SMOTE()
        t_data_X = t_data.drop('Class', axis = 1)
        t_data_y = t_data.loc[:, ['Class']]

        t_data_X, t_data_y = oversample.fit_resample(t_data_X, t_data_y)
        t_data = pd.concat([t_data_X, t_data_y], axis = 1)
        t_data = t_data.sample(frac = 1, random_state = 42, ignore_index = True)
    elif balancing_method == "under":
            t_data_pos = t_data.loc[t_data["Class"] == 1]
            t_data_neg = t_data.loc[t_data["Class"] == 0]

            pos_class_size = len(t_data_pos)

            t_subsample_neg = t_data_neg.sample(n = pos_class_size, random_state = 42, ignore_index = True)
            t_data = pd.concat([t_data_pos, t_subsample_neg], axis = 0)
            t_data = t_data.sample(frac = 1, random_state = 42, ignore_index = True)        
    else:
        raise Exception("Did not specify sampling method.")
    
    return t_data

def custom_kfold(data, cv_n, balancing_method, model, random_state = None):

    scores = {
        "train": {
            "accuracy_lst": [],
            "precision_lst": [],
            "recall_lst": [],
            "f1_lst": []
        },
        "val": {
            "accuracy_lst": [],
            "precision_lst": [],
            "recall_lst": [],
            "f1_lst": []
        }
    }

    fold_boundaries = get_fold_boundaries(data_input = data, cv_n = cv_n, random_state = random_state)

    for i, (fold_start, fold_end) in enumerate(tqdm(fold_boundaries)):

        # Getting the data
        t_data = get_tdata(data_input = data, fold_boundaries = fold_boundaries, fold_index = i)
        v_data = data.iloc[fold_start:fold_end + 1]
        
        # Scaling
        robustScaler = RobustScaler()
        t_data, v_data = scale_data(t_data, v_data, robustScaler, sel_cols = ['Time', 'Amount'])

        # Data Balancing
        t_data = data_balancing(t_data, balancing_method = balancing_method)

        # Training
        X_train = t_data.drop('Class', axis = 1)
        y_train = t_data.loc[:, 'Class']

        model.fit(X_train, y_train)
        best_model = model.best_estimator_

        y_train_pred = best_model.predict(X_train)
        # y_train_score = best_model.predict_proba(X_train)[:, 1]

        t_accuracy = accuracy_score(y_train, y_train_pred)
        t_precision = precision_score(y_train, y_train_pred)
        t_recall = recall_score(y_train, y_train_pred)
        # t_precision_curve, t_recall_curve, _ = precision_recall_curve(y_train, y_train_score)
        t_f1 = f1_score(y_train, y_train_pred)

        # Testing
        X_val = v_data.drop('Class', axis = 1)
        y_val = v_data.loc[:, 'Class']

        y_val_pred = best_model.predict(X_val)
        # y_val_score = best_model.predict_proba(X_val)[:, 1]

        v_accuracy = accuracy_score(y_val, y_val_pred)
        v_precision = precision_score(y_val, y_val_pred)
        v_recall = recall_score(y_val, y_val_pred)
        # v_precision_curve, v_recall_curve, _ = precision_recall_curve(y_val, y_val_score)
        v_f1 = f1_score(y_val, y_val_pred)

        
        scores["train"]["accuracy_lst"].append(t_accuracy)
        scores["val"]["accuracy_lst"].append(v_accuracy)

        scores["train"]["precision_lst"].append(t_precision)
        scores["val"]["precision_lst"].append(v_precision)

        scores["train"]["recall_lst"].append(t_recall)
        scores["val"]["recall_lst"].append(v_recall)

        scores["train"]["f1_lst"].append(t_f1)
        scores["val"]["f1_lst"].append(v_f1)

    
    scores["train"]["accuracy_lst"] = np.mean(scores['train']['accuracy_lst'])
    scores["val"]["accuracy_lst"] = np.mean(scores['val']['accuracy_lst'])

    scores["train"]["precision_lst"] = np.mean(scores['train']['precision_lst'])
    scores["val"]["precision_lst"] = np.mean(scores['val']['precision_lst'])

    scores["train"]["recall_lst"] = np.mean(scores['train']['recall_lst'])
    scores["val"]["recall_lst"] = np.mean(scores['val']['recall_lst'])

    scores["train"]["f1_lst"] = np.mean(scores["train"]["f1_lst"])
    scores["val"]["f1_lst"] = np.mean(scores["val"]["f1_lst"])
    
    return scores, best_model
