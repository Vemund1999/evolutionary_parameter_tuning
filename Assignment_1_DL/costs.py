from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import xgboost as xgb

from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.class_weight import compute_class_weight

from collections import Counter

from dataset import Dataset
from data_pipeline import DataPipeline
import matplotlib.pyplot as plt


from sklearn.base import clone



# Load dataset
x, y = Dataset().alt_get_dataset()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
# X_train = X_full_train[:150000]
# y_train = y_full_train[:150000]

# Transform the dataset using a pipeline
pipeline = DataPipeline()
transformer_x_train = pipeline.get_transformer(X_train)
x_train = transformer_x_train.fit_transform(X_train)
x_test = transformer_x_train.transform(X_test)

def get_matrix_values(model, params):
    model.set_params(**params)
    model.fit(x_train, y_train)

    total_rows = len(x_test)

    y_pred = model.predict(x_test)

    cm = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = cm.ravel()

    TN = TN / total_rows * 100  # TN as percentage of all rows
    FP = FP / total_rows * 100  # FP as percentage of all rows
    FN = FN / total_rows * 100  # FN as percentage of all rows
    TP = TP / total_rows * 100  # TP as percentage of all rows

    return TN, FP, FN, TP

# Get overall TN, FP, FN, TP values

def vary_class_weights(model, params):
    results = {}
    total_rows = len(x_test)
    model.set_params(**params)
    for i in range(11):
        shift_up = 0.0 + i / 10
        shift_down = 1.0 - i / 10

        instance_weights = [shift_up if label == 0 else shift_down for label in y_train]

        cloned = clone(model)
        cloned.fit(x_train, y_train, sample_weight=instance_weights)

        y_pred = cloned.predict(x_test)

        cm = confusion_matrix(y_test, y_pred)
        TN, FP, FN, TP = cm.ravel()

        results[i] = [
            TN / total_rows * 100,  # TN as percentage of all rows
            FP / total_rows * 100,  # FP as percentage of all rows
            FN / total_rows * 100,  # FN as percentage of all rows
            TP / total_rows * 100   # TP as percentage of all rows
        ]
        results[f"{i}_weights"] = [shift_up, shift_down]  # Store weights as floats

    return results




def plot_results(results, TN, FP, FN, TP, file_name):
    # Extract confusion matrix values for plotting
    iterations = list(range(11))  # Using range to match iteration numbers
    TN_values = [results[i][0] for i in iterations]  # True Negatives
    FP_values = [results[i][1] for i in iterations]  # False Positives
    FN_values = [results[i][2] for i in iterations]  # False Negatives
    TP_values = [results[i][3] for i in iterations]  # True Positives

    # Extract the weight values for the x-axis labels
    weight_labels = [f"({results[f'{i}_weights'][0]:.1f}, {results[f'{i}_weights'][1]:.1f})" for i in iterations]

    # Create a figure and axis for plotting
    plt.figure(figsize=(10, 6))

    # Plot each component of the confusion matrix on the same graph
    plt.plot(iterations, TN_values, label='True Negatives (TN)', marker='o')
    plt.plot(iterations, FP_values, label='False Positives (FP)', marker='o')
    plt.plot(iterations, FN_values, label='False Negatives (FN)', marker='o')
    plt.plot(iterations, TP_values, label='True Positives (TP)', marker='o')

    # Add horizontal lines for TN, FP, FN, TP values
    plt.axhline(y=TN, color='blue', linestyle='--', label='Default TN %')
    plt.axhline(y=FP, color='orange', linestyle='--', label='Default FP %')
    plt.axhline(y=FN, color='red', linestyle='--', label='Default FN %')
    plt.axhline(y=TP, color='green', linestyle='--', label='Default TP %')

    # Add labels and title
    plt.xlabel('Weights of label values - (0, 1)')
    plt.ylabel('Confusion Matrix Values (%)')
    plt.title('Confusion Matrix Values vs Weights Shift')

    # Add custom x-ticks with weight labels
    plt.xticks(iterations, weight_labels, rotation=45)  # Rotate labels for better readability

    plt.legend()  # Show a legend to differentiate the lines
    plt.grid(True)

    # Save the plot to a file
    plt.tight_layout()  # Adjust layout to prevent label clipping
    plt.savefig("costs/" + file_name)






# Assuming results are generated with the someting() function
params = [
    # XGBoost
    # GA
    {'eta': 0.650526310496016, 'gamma': 0.700417268555284, 'n_estimators': 1056, 'max_depth': 3, 'min_child_weight': 1, 'max_delta_step': 4, 'subsample': 0.836867407543178, 'sampling_method': 'uniform', 'colsample_bytree': 0.9698688251745518, 'colsample_bylevel': 0.8438854659636649, 'colsample_bynode': 0.500193527056886, 'lambda': 12, 'alpha': 7, 'tree_method': 'hist', 'refresh_leaf': 1, 'max_leaves': 382},
    # SEQ
    {'eta': 0.3, 'gamma': 1.0, 'n_estimators': 250},
    # DEF
    {},

    # Random forest
    # GA
    {'n_estimators': 1923, 'criterion': 'log_loss', 'max_depth': 150, 'min_samples_split': 10, 'min_samples_leaf': 5, 'max_leaf_nodes': 6, 'max_samples': 982},
    # SEQ
    {'n_estimators': 100, 'criterion': 'entropy', 'max_depth': 50, 'min_samples_split': 3},
    # DEF
    {}
]

models = [
    xgb.XGBClassifier(),
    xgb.XGBClassifier(),
    xgb.XGBClassifier(),
    RandomForestClassifier(),
    RandomForestClassifier(),
    RandomForestClassifier()
]

names = [
    "GA xgb",
    "SEQ xgb",
    "DEF xgb",

    "GA rf",
    "SEQ rf",
    "DEF rf"
]

for i in range(len(models)):
    print(names[i])
    TN, FP, FN, TP = get_matrix_values(models[i], params[i])
    results = vary_class_weights(models[i], params[i])
    plot_results(results, TN, FP, FN, TP, names[i] + ".png")






