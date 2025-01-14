from logging import raiseExceptions, exception

import xgboost as xgb
import random
import math

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

from data_pipeline import DataPipeline
from dataset import Dataset

import time


import os

class ParameterOptimizer:
    def __init__(self,
                 model,
                 X_train, X_test, y_train, y_test,
                 iterations,
                 pop_size,
                 param_settings,
                 mutation_rate,
                 proportionate_roulette,
                 name_of_files,
                 device = 'cpu',
                 main_metric = "accuracy", second_metric = "r1"
                 ):

        self.model = model

        self.main_metric = main_metric
        self.second_metric = second_metric





        self.device = device

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.iterations = iterations
        self.pop_size = pop_size
        if self.pop_size % 2 != 0:
            raise Exception(f"pop_size must be divisible by 2. the pop_size is {self.pop_size}") # ctrl+f "selection" for å se hvorfor

        self.param_settings = param_settings

        self.mutation_rate = mutation_rate
        if not isinstance(self.mutation_rate, int) or self.mutation_rate < 0 or self.mutation_rate > len(self.param_settings):
            raise Exception("Mutation rate must be a positive integer that is lower or equal to the amounts of params")

        self.proportionate_roulette = proportionate_roulette

        self.name_of_files = name_of_files

        self.best_params = None
        self.best_fitness = None
        self.best_other_metric = None



    def make_graphs(self, iter_stats):

        # create folder if it dosent exist
        os.makedirs(self.name_of_files, exist_ok=True)


        # PLOTTER FITNESS OVER GENERASJONER ===
        regular_stats = {k: v for k, v in iter_stats.items() if not (isinstance(k, str) and (
                    k.endswith('_pop') or k.endswith('_params') or k.endswith('_params_fitnesses')))}

        # Plot highest fitness over iterations
        highest_fitness_values = [value[0] for value in regular_stats.values()]
        iterations = list(regular_stats.keys())  # ERROR
        plt.plot(iterations, highest_fitness_values, marker='o')
        plt.xlabel('Iteration')
        plt.ylabel(f'Highest {self.main_metric}')
        plt.title(f'Highest {self.main_metric} Over Iterations')
        plt.savefig(f"{self.name_of_files}_fitness_over_time.png")
        plt.close()  # Close the figure to avoid overlap

        # Plot highest other_metric over iterations
        highest_other_metric = [value[2] for value in regular_stats.values()]
        plt.plot(iterations, highest_other_metric, marker='o')
        plt.xlabel('Iteration')
        plt.ylabel(f'Highest {self.second_metric}')
        plt.title(f'Highest {self.second_metric} Over Iterations')
        plt.savefig(f"{self.name_of_files}_accuracy_over_time.png")
        plt.close()  # Close the figure to avoid overlap


        #
        # Now handle population-level data (like '0_pop', '1_pop')
        pop_stats = {k: v for k, v in iter_stats.items() if isinstance(k, str) and k.endswith('_pop')}

        # Extract the iteration indices
        iterations = list(pop_stats.keys())

        # Initialize lists to store fitness and accuracy values for plotting
        fitness_values = []
        other_metric_values = []

        # Iterate over the iterations to collect fitness and accuracy values
        for i in iterations:
            fitness_values.append(pop_stats[i][0])  # Fitness values (list of fitness for each individual)
            other_metric_values.append(pop_stats[i][1])  # Accuracy values (list of other_metric for each individual)

        # For simplicity, let's take the average fitness and accuracy per iteration
        avg_fitness = [sum(f) / len(f) for f in fitness_values]  # Average fitness for each iteration
        avg_accuracy = [sum(a) / len(a) for a in other_metric_values]  # Average accuracy for each iteration

        # Plot individual and average fitness values
        plt.figure(figsize=(10, 5))  # Create a new figure

        # Plot individual fitness values (blue dots)
        for i, fitness_list in enumerate(fitness_values):
            plt.scatter([i] * len(fitness_list), fitness_list, color='blue',
                        label='Individual Fitness' if i == 0 else "", alpha=0.6)

        # Plot average fitness values (red dots)
        plt.plot(range(len(avg_fitness)), avg_fitness, 'ro', label=f'Average {self.main_metric}', markersize=8)

        # Add labels and title
        plt.xlabel('Iterations')  # X-axis label
        plt.ylabel(f'{self.main_metric}'.title())  # Y-axis label
        plt.title(f'{self.main_metric} over Iterations'.title())  # Title of the plot
        plt.legend()  # Show legend
        plt.grid(True)  # Show grid lines
        plt.savefig(f"{self.name_of_files}_population_fitness.png")  # Save the plot
        plt.close()  # Close the figure to avoid overlap

        # Plot individual and average accuracy values
        plt.figure(figsize=(10, 5))  # Create a new figure

        # Plot individual accuracy values (blue dots)
        for i, accuracy_list in enumerate(other_metric_values):
            plt.scatter([i] * len(accuracy_list), accuracy_list, color='blue',
                        label=f'Individual {self.second_metric}'.title() if i == 0 else "", alpha=0.6)

        # Plot average accuracy values (red dots)
        plt.plot(range(len(avg_accuracy)), avg_accuracy, 'ro', label=f'Average {self.second_metric}'.title(), markersize=8)

        # Add labels and title
        plt.xlabel('Iterations')  # X-axis label
        plt.ylabel(f'{self.second_metric}'.title())  # Y-axis label
        plt.title(f'{self.second_metric} over Iterations'.title())  # Title of the plot
        plt.legend()  # Show legend
        plt.grid(True)  # Show grid lines
        plt.savefig(f"{self.name_of_files}_population_accuracy.png")  # Save the plot
        plt.close()  # Close the figure








        # Separate parameter data (keys ending with '_params')
        params_stats = {k: v for k, v in iter_stats.items() if isinstance(k, str) and k.endswith('_params')}
        fitnesses_params_stats = {k: v for k, v in iter_stats.items() if isinstance(k, str) and k.endswith('_params_fitnesses')}

        # Initialize a set to track all unique keys across all dictionaries
        all_keys = set()

        # First pass: Collect all unique keys in the dictionaries
        for param_list in params_stats.values():
            for param_dict in param_list:
                all_keys.update(param_dict.keys())

        # Now for each key, we will plot its value over time
        for key_to_plot in all_keys:
            # Initialize lists to hold the x (iterations) and y (values for the current key)
            iterations = []
            all_values = []
            highest_fitness_indices = []  # To store indices of highest fitness individuals

            # Iterate through each iteration (like '0_params', '1_params', etc.)
            for iteration_key, param_list in params_stats.items():
                iteration = int(iteration_key.split('_')[0])  # Extract iteration number
                iterations.append(iteration)

                # Get the corresponding fitness list for this iteration
                fitness_list = fitnesses_params_stats.get(f"{iteration}_params_fitnesses", [])

                # Find the index of the highest fitness individual
                if fitness_list:
                    highest_fitness_index = fitness_list.index(max(fitness_list))
                else:
                    highest_fitness_index = None

                # Collect the values of the current key (`key_to_plot`) from each dictionary in this iteration's param_list
                values_for_key = [d.get(key_to_plot, None) for d in param_list]
                all_values.append(values_for_key)
                highest_fitness_indices.append(highest_fitness_index)

            # Plot the values of the current key
            plt.figure(figsize=(10, 6))

            # Scatter plot the values for each iteration
            for i, values in enumerate(all_values):
                if ('gpu' in values) or ('cpu' in values):
                    continue
                iteration_number = iterations[i]
                # Scatter plot for all values of the key at the current iteration (as blue dots)
                plt.scatter([iteration_number] * len(values), values, color='blue', alpha=0.6)

                # Highlight the individual with the highest fitness in red
                if highest_fitness_indices[i] is not None:
                    # Get the value of the key for the highest fitness individual
                    highest_fitness_value = values[highest_fitness_indices[i]]
                    plt.scatter(iteration_number, highest_fitness_value, color='red',
                                label=f'Highest {self.main_metric}' if i == 0 else "", s=100)
            # Add labels and title
            plt.xlabel('Iterations')
            plt.ylabel(f'Values of {key_to_plot}')
            plt.title(f'Values of {key_to_plot} over Iterations (Highest Fitness in Red)')
            plt.grid(True)
            plt.legend()

            # Save each plot with the key name in the filename
            plt.savefig(f"{self.name_of_files}_param_values_{key_to_plot}_over_time_with_fitness.png")
            plt.close()  # Close the plot to prevent overlap with the next one








    def run(self):


        parent_params = self.generate_initial_pop()
        fitness_parents, other_metric_parents = self.determine_fitness(parent_params)
        iter_stats = {}
        for i in range(self.iterations):

            children_params = self.selection(parent_params, fitness_parents)
            fitness_children, other_metric_children = self.determine_fitness(children_params)
            fitness_parents, parent_params, other_metric_parents = self.survivor_selection(fitness_parents, fitness_children,
                                                                                           parent_params, children_params,
                                                                                           other_metric_parents, other_metric_children)

            iter_stats[i] = [max(fitness_parents), len(parent_params)-1, max(other_metric_parents)] # Of intrest
            iter_stats[f"{i}_pop"] = [fitness_parents, other_metric_parents]
            iter_stats[f"{i}_params"] = [{k: v for k, v in d.items() if (k != 'gpu' or k != 'cpu')} for d in parent_params]
            iter_stats[f"{i}_params_fitnesses"] = fitness_parents

            i += 1
            print(f"Iteration: {i}/{self.iterations}, highest fitness: {max(fitness_parents)}, highest other param: {max(other_metric_parents)}")


        self.make_graphs(iter_stats)

        self.best_params = parent_params[0]
        self.best_fitness = fitness_parents[0]
        self.best_other_metric = other_metric_parents[0]








    def survivor_selection(self, fitness_parents, fitness_children, parent_params, children_params, other_metric_parents, other_metric_children):

        fitnesses = fitness_parents + fitness_children
        params = parent_params + children_params
        other_metric = other_metric_parents + other_metric_children

        combined = list(zip(fitnesses, params, other_metric))
        sorted_combined = sorted(combined, key=lambda x: x[0], reverse=True)
        sorted_fitnesses, sorted_params, sorted_other_metric = zip(*sorted_combined)

        selected_fitnesses = list(sorted_fitnesses)[:-self.pop_size]
        selected_params = list(sorted_params)[:-self.pop_size]
        selected_other_metric = list(sorted_other_metric)[:-self.pop_size]

        return selected_fitnesses, selected_params, selected_other_metric




    def generate_initial_pop(self):
        parents_params = []
        for i in range(self.pop_size):
            params = {}
            for key, value in self.param_settings.items():
                if isinstance(value[0], int):
                    params[key] = random.randint(value[0], value[1])
                elif isinstance(value[0], float):
                    params[key] = random.uniform(value[0], value[1])
                else: #isinstance(value[0], str)
                    params[key] = random.choice(self.param_settings[key])
            parents_params.append(params)

            # gpu
            if 'device' in self.model.get_params():
                params['device'] = self.device

        return parents_params



    def determine_fitness(self, parent_params):
        other_metrics = []
        parent_fitnesses = []
        for i in range(len(parent_params)):
            clf = self.model
            clf.set_params(**parent_params[i])
            clf.fit(self.X_train, self.y_train)
            y_pred = clf.predict(self.X_test)

            # ==========
            acc = accuracy_score(self.y_test, y_pred)

            cm = confusion_matrix(self.y_test, y_pred)
            TN, FP, FN, TP = cm.ravel()
            TPR = TP / (TP + FN)
            TNR = TN / (FP + TN)
            res = math.sqrt(TPR + TNR)

            parent_fitnesses.append(acc)
            other_metrics.append(res)
            # ==========

        return parent_fitnesses, other_metrics
    


    def choose_parents_to_mate(self, parent_params, fitnesses):

        if self.proportionate_roulette:
            total_fitness = sum(fitnesses)
            probabilities = [fitness / total_fitness for fitness in fitnesses]
            cumulative_distribution = np.cumsum(probabilities)

            parents_to_mate = []
            matched_params = {}
            while len(parents_to_mate) <= self.pop_size-2:
                parents_added = 0
                first_mate = None
                while True:
                    random_value = np.random.rand()
                    selected_index = np.searchsorted(cumulative_distribution, random_value)

                    if selected_index not in matched_params:
                        matched_params[selected_index] = []

                    if parents_added == 1 and first_mate in matched_params[selected_index] and selected_index in matched_params[first_mate]:
                        continue

                    if parents_added == 1:
                        matched_params[selected_index].append( first_mate )
                        matched_params[first_mate].append( selected_index )
                        parents_to_mate.append( parent_params[selected_index] )
                        break

                    if parents_added == 0:
                        first_mate = selected_index
                        parents_to_mate.append( parent_params[first_mate] )
                        parents_added += 1

            return parents_to_mate


        if not self.proportionate_roulette:


            zipped = list(zip(fitnesses, parent_params))
            zipped.sort(key=lambda x: x[0], reverse=True)  # Sort by fitness value

            sorted_fitnesses, sorted_parent_params = zip(*zipped)
            sorted_parent_params = list(sorted_parent_params)


            # generate probabilites using linear decay
            n = len(parent_params) + 1 # +1 to ensure the worst parent also has a chance
            probabilities = [self.max_p_for_highest_fit - (self.max_p_for_highest_fit / (n - 1)) * i for i in range(n)]



            highest_fit = self.max_p_for_highest_fit
            next_highest = ( self.max_p_for_highest_fit / 3 ) * 2
            lowest_fit = self.max_p_for_highest_fit / 3
            max_p = lowest_fit + next_highest + highest_fit

            # top fit har 50% for å bli plukket
            # 1/3 av de neste øvre har 33% for å bli plukket
            # nedre 2/3 har 16% for å bli plukket

            # bruk probability
            # foreldre er matches i par på to-og-to ... [1,2   ,   3,4   ,    5,6]


            parents_to_mate = []
            parents_to_mate_indexes = []
            has_mates = {}
            while len(parents_to_mate) <= self.pop_size-2:
                first_mate = None

                added_for_now = 0
                i = 0
                while added_for_now < 2:
                    spin_pick = random.uniform(0, max_p)
                    picked_mate_index = None

                    if spin_pick >= highest_fit: # highest pick
                        picked_mate_index = 0

                    elif lowest_fit < spin_pick < highest_fit: # 2nd highest pick
                        choose_from = int(self.pop_size / 3) + 1 # +1 to not include highest fit
                        index = random.randint(1, choose_from)
                        picked_mate_index = index

                    elif spin_pick <= lowest_fit:
                        choose_from = int(self.pop_size / 3) + 1
                        index = random.randint(choose_from, self.pop_size-1)
                        picked_mate_index = index


                    if picked_mate_index not in has_mates:
                        has_mates[picked_mate_index] = []

                    if  added_for_now == 1:
                        if (first_mate == picked_mate_index) or (picked_mate_index in has_mates[first_mate]) or (first_mate in has_mates[picked_mate_index]):
                            i += 1
                            if i == 1000:
                                raise Exception(f"stuck at {picked_mate_index}. This might be caused by too high p value.")
                            continue
                        has_mates[picked_mate_index].append(first_mate)
                        has_mates[first_mate].append(picked_mate_index)

                    first_mate = picked_mate_index
                    added_for_now += 1
                    parents_to_mate.append( sorted_parent_params[picked_mate_index] )
                    parents_to_mate_indexes.append( picked_mate_index )
            return parents_to_mate






    def selection(self, parent_params, fitnesses):

        parents_to_mate = self.choose_parents_to_mate(parent_params, fitnesses)

        children_params = []
        for p in range(0, len(parents_to_mate), 2):
            mates = parents_to_mate[ p : p+2 ] # pop_size må være delelig med 2
            child_1_params = {}
            child_2_params = {}

            # (param_settings) - 1  ... because it shouldn't pick the 'device' param.
            params_to_mutate = sorted(random.sample(range(0, len(self.param_settings)-1), self.mutation_rate))

            for param_key, param_1_value, param_2_value, i in zip(mates[0], mates[0].values(), mates[1].values(), range(len(self.param_settings)-1)): # same for 'device' here

                if i % 2 == 0:
                    child_1_params[param_key] = param_1_value
                    child_2_params[param_key] = param_2_value
                else:
                    child_1_params[param_key] = param_2_value
                    child_2_params[param_key] = param_1_value

                if i in params_to_mutate:
                    child_1_params = self.mutate(child_1_params, param_key)
                    child_2_params = self.mutate(child_2_params, param_key)

            children_params.append(child_1_params)
            children_params.append(child_2_params)

        return children_params
    



                    
    def mutate(self, child, param):
        # if parameter value is a string, change to a random other string
        if isinstance(self.param_settings[param][0], str):
            child[param] = random.choice(self.param_settings[param])
            return child

        if param == 'refresh_leaf':
            if child[param] == 0:
                child[param] = 1
            else:
                child[param] = 0
            return  child

        # if parameter value is an integer or a float, increment/decrement by random value
                    # parameter               # multiplying by random     # adding or subtracting
        jump = self.param_settings[param][2] * random.choice([1, 3]) * random.choice([1, -1])

        # checking if adding or subtracting will go below min value or above max value, if so, do opposite operation
        if (child[param] + jump) < self.param_settings[param][0] or (child[param] + jump) > self.param_settings[param][1]:
            jump = -jump
        child[param] = child[param] + jump

        return child






# dataset
x, y = Dataset().alt_get_dataset()
X_full_train, X_test, y_full_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
X_train = X_full_train[:150000]
y_train = y_full_train[:150000]

# apply pipeline
pipeline = DataPipeline()
transformer_x_train = pipeline.get_transformer(X_train)  # Adjusted to accept the feature to remove
x_train = transformer_x_train.fit_transform(X_train)
x_test = transformer_x_train.transform(X_test)




# define parameters
# 17 params - 1 kommentert ut
xgb_params = {
    # min val    # max val     # stepper for mutation
    'eta': [0.01, 1, 0.03],
    'gamma': [0.01, 1.0, 0.03],
    'n_estimators': [10, 1500, 100],
    'max_depth': [1, 10, 1],
    'min_child_weight': [1, 5, 1],
    'max_delta_step': [1, 5, 1],
    'subsample': [0.01, 1.0, 0.03],
    'sampling_method': ['uniform'],
    'colsample_bytree': [0.01, 1.0, 0.03],
    'colsample_bylevel': [0.01, 1.0, 0.03],
    'colsample_bynode': [0.01, 1.0, 0.03],
    'lambda': [0, 20, 2],
    'alpha': [0, 20, 2],
    'tree_method': ['hist'],
    'refresh_leaf': [0, 1, 1],
    'max_leaves': [0, 500, 50],
    'max_bin': [100, 500, 50]
}

rf_params = {
    'n_estimators': [100, 3000, 100], #===
    'criterion': ['gini', 'entropy', 'log_loss'], #===
    'max_depth': [1, 500, 2], #===
    'min_samples_split': [2, 15, 1], #===
    'min_samples_leaf': [1, 14, 2], #===
    'max_leaf_nodes': [2, 30, 2], #===
    'max_samples': [150, 1000, 150]  #===
}

# models
folder = "rf/"
rf = RandomForestClassifier()
xgb = xgb.XGBClassifier()





# using models

# ==============================
# XGB
# =============================
from scipy import stats
import statistics

def get_best_acc_and_params(accuracies, params):
    temp_i = 0
    for i in range(len(accuracies)):
        if  accuracies[i] > accuracies[temp_i]:
            temp_i = i
    best_acc = accuracies[temp_i]
    best_params = params[temp_i]

    return best_acc, best_params, statistics.mean(accuracies)


#low_pop_accuracies = []
#low_pop_params = []

high_pop_accuracies = []
high_pop_params = []

# using optimizer
for i in range(10):
    high_folder = f"new_evo_runs/rf/high_pop/{i}/"

    # fører resultater

    high_pop = ParameterOptimizer(
                                        rf,
                                        x_train, x_test, y_train, y_test,
                                        5,
                                        20,
                                        rf_params,
                                        4,
                                        True,
                                        high_folder,
                                        'cpu',
                                        "accuracy", "r1"
                                        )
    high_pop.run()
    high_pop_accuracies.append( high_pop.best_fitness )
    high_pop_params.append( high_pop.best_params )



print(high_pop_accuracies)

# finding best model of all experiments, the mean, and printing all results
print("High pop, low it")
high_best_acc, high_best_params, high_mean_acc = get_best_acc_and_params(high_pop_accuracies, high_pop_params)
print(f"Best accuracy: {high_best_acc}, mean accuracy: {high_mean_acc}, best params: {high_best_params}")
print("All best accuracies and params found: ")
for i in range(len(high_pop_accuracies)):
    print(f"accuracy: {high_pop_accuracies[i]}, params: {high_pop_params[i]}")
print("")



