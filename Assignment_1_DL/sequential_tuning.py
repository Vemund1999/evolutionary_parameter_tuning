
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score


from dataset import Dataset
from data_pipeline import DataPipeline






def get_dataset():
    x, y = Dataset().alt_get_dataset()
    X_full_train, X_test, y_full_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
    X_train = X_full_train[:150000]
    y_train = y_full_train[:150000]

    pipeline = DataPipeline()
    transformer_x_train = pipeline.get_transformer(X_train)  # Adjusted to accept the feature to remove
    x_train = transformer_x_train.fit_transform(X_train)
    x_test = transformer_x_train.transform(X_test)

    return x_train, x_test, y_train, y_test






def sequential_parameter_tuning(model, param_settings, x_train, x_test, y_train, y_test):
    stat = {}

    params_found_so_far = {}
    best_accuracy = 0
    iterations = 0

    tried_combos = 0

    for param, values in param_settings.items():
        print("")
        print("so far:")
        print(params_found_so_far)
        print(f"accuracy: {best_accuracy}")
        print(f"checking param: {param}")
        print("")

        for value in values:
            tried_combos += 1
            iterations += 1

            param_value_to_try = {param: value}
            model.set_params(**{**param_value_to_try, **params_found_so_far})
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)

            print(f"current acc on param: {accuracy}, {param}: {value}")
            if best_accuracy < accuracy:
                best_accuracy = accuracy
                params_found_so_far[param] = value
                stat[f'{param}___{value}'] = best_accuracy

    print(f"tried combos: {tried_combos}")
    return stat



import matplotlib.pyplot as plt

def plot_sequential(stat, filename, title):
    # Extract the parameter names and accuracies
    x_labels = []
    y_values = []

    for key, accuracy in stat.items():
        param, value = key.split('___')
        x_labels.append(f"{param}: {value}")  # Format as "param: value"
        y_values.append(accuracy)

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(x_labels, y_values, marker='o', linestyle='-', color='b')
    plt.xticks(rotation=45, ha='right')  # Rotate x labels for better visibility
    plt.xlabel('Parameter Settings')
    plt.ylabel('Best Model Accuracy')
    plt.title(title)
    plt.grid()
    plt.tight_layout()  # Adjust layout to make room for x-axis labels
    plt.savefig(filename + ".png")


x_train, x_test, y_train, y_test = get_dataset()


xgb_params_seq = {
    'eta': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'gamma': [0.0, 0.01, 1.0, 5.0, 10.0, 20.0],
    'n_estimators': [100, 250, 500, 1000, 1500],
    'max_depth': [6, 7, 8, 9, 10],
    'min_child_weight': [1, 2,3,4,5],
    'max_delta_step': [0, 1,2,3,4,5],
    'subsample': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'sampling_method': ['uniform'],
    'colsample_bytree': [1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'colsample_bylevel': [1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'colsample_bynode': [1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'lambda': [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    'alpha': [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    'tree_method': ['hist'],
    'refresh_leaf': [1, 0],
    'max_leaves': [0, 50, 100, 150, 200, 250, 300, 400, 500],
    'max_bin': [256, 100, 150, 200, 250, 300, 400, 500]
}

rf_params_seq = {
    'n_estimators': [100, 300, 350, 750, 850, 900, 1100, 1200, 1700, 1900, 1900, 2400, 2700, 3000],
    'criterion': ['gini', 'entropy', 'log_loss'], #===
    'max_depth': [None, 25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400, 450, 500], #===
    'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15], #===
    'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], #===
    'min_weight_fraction_leaf': [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5], #===
    'max_leaf_nodes': [None, 2, 3, 5, 6, 8, 9, 12, 15, 18, 21, 24, 27, 30], #===
    'max_samples': [None, 67, 150, 245, 312, 475, 550, 620, 738, 812, 875, 920, 990, 1000]  #===
}

def default():
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"default accuracy: {accuracy}")
default()

xgb = xgb.XGBClassifier()
rf = RandomForestClassifier(random_state=42)



print("xgboost")
stat = sequential_parameter_tuning(rf, rf_params_seq, x_train, x_test, y_train, y_test)
plot_sequential(stat, "seq_and_rand_opt/seq_rf", "Sequential Search Early Break XGBoost")
















