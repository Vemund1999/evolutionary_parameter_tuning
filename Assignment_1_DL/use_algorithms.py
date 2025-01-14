import time
from random import randint

from joblib.externals.cloudpickle import instance
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import xgboost as xgb
import random
from sklearn.metrics import confusion_matrix
import math

from data_pipeline import DataPipeline
from dataset import Dataset

# Load dataset
dataset = Dataset().get_dataset()
n = len(dataset)
fraction = 100000 / n
_, dataset_subset = train_test_split(dataset, test_size=fraction, random_state=42)

target = Dataset().target
X_train, X_test, y_train, y_test = train_test_split(dataset_subset, dataset_subset[target], test_size=0.25, random_state=42)

# Create column transformer
transformer = DataPipeline().create_column_transformer()
X_train = transformer.fit_transform(X_train)
X_test = transformer.fit_transform(X_test)









params = {
    #'n_estimators': 100,       # Number of boosting rounds (trees)
    'max_bin': 256,      # Learning rate (shrinkage)
    'max_depth': 6,             # Maximum depth of a tree
    # 'subsample': 0.8,          # Subsample ratio of the training instances
    # 'colsample_bytree': 0.8,   # Subsample ratio of columns when constructing each tree
     'gamma': 0                # Minimum loss reduction required to make a further partition
    # 'reg_alpha': 0,            # L1 regularization term on weights
    # 'reg_lambda': 1,           # L2 regularization term on weights
    # 'scale_pos_weight': 1,     # Balancing of positive and negative weights
    # 'objective': 'binary:logistic',  # Specify the learning task and the objective
    # 'random_state': 42         # Random seed for reproducibility
}


parents_params = []
# generating parents
nr_of_parents = 10
for p in range(nr_of_parents):

    param_settings = {
                # min val    # max val     # stepper for mutation
        'gamma': [0.01,      1.0,           0.03],
        'n_estimators': [10, 1500, 100],
        'max_depth': [1, 10, 1],
        'min_child_weight': [1, 5, 1],
        'max_delta_step': [1, 5, 1],
        'subsample': [0.01, 1.0, 0.03],
    }


    params = {}
    for key, value in param_settings.items():
        if instance(value[0], int):
            params[key] = random.randint(value[0], value[1])
        else: # instance(value[0], float)
            params[key] = random.uniform(0.01, 1.0)
        parents_params.append(params)


# selection
# putting all parent fitnesses in a list
parent_fitnesses = {}
for i in range(len(parents_params)):
    clf = xgb.XGBClassifier(**parents_params[i])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = cm.ravel()
    TPR = TP / (TP + FN)
    TNR = TN / (FP + TN)
    fitness = math.sqrt( TPR + TNR )
    parent_fitnesses[i] = fitness

new_parents = []
# sorting the parents after their fitness
sorted_desc_dict = dict(sorted(parent_fitnesses.items(), key=lambda item: item[1], reverse=True))
# excluding the parents with the two lowest fitnesses
keys = list(sorted_desc_dict.keys())
for key in keys[:-2]:
    new_parents.append( parents_params[key] )
parents_params = new_parents




# crossover
children_params = []
nr_of_children = 10
i = 0
for p in range(nr_of_parents, 2):
    mates = parents_params[p*2:p*2+1]
    child_1_params = {}
    child_2_params = {}

    param_to_mutate = random.randint(0, 6, 1)
    counter = 0
    for param_key, param_1_value, param_2_value, i in zip(mates[0], mates[0].values(), mates[1].values(), range(3)):
        if (counter == 0):
            child_1_params[param_key] = param_1_value
            child_2_params[param_key] = param_2_value
        else: # counter == 1
            child_1_params[param_key] = param_2_value
            child_2_params[param_key] = param_1_value
        counter += 1
        if (counter) == 2:
            counter = 0


        # mutation
        if i == param_to_mutate:
                    # min value to jump            # multiply by random     # add or subtract
            jump = param_settings[param_key][2] * random.choice([1, 3]) * random.choice([1, -1])

            temp_1 = child_1_params[param_key] + jump
            temp_2 = child_2_params[param_key] + jump
            # if new value is less than min - rather add it, if new value is larger than max - rather subtract it

            if temp_1 < parents_params[param_key][0] or temp_1 > parents_params[param_key][1]:
                jump * -1
            if temp_2 < parents_params[param_key][0] or temp_2 > parents_params[param_key][1]:
                jump * -1

            child_1_params[param_key] = temp_1
            child_2_params[param_key] = temp_2

            children_params.append(child_1_params)
            children_params.append(child_2_params)

# replace parent population with child population
parents_params = children_params


