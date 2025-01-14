# About project
This project showcases use of a simple genetic algorithm for parameter uning of XGBoost and Random Forest from scikit-learn.

The following genetic algorithm was used:

![bilde](https://github.com/user-attachments/assets/f6bbd14a-1c4b-43bf-9cc3-f851a3a1d814)

Each individual is a parameter combination. Each gene is different a parameter value.


# Dataset
Loan defaulter classification was used to for the tuning algorithm.

https://www.kaggle.com/datasets/prateek146/loan-defaulter-classification?select=train_indessa.csv

The goal of the dataset is to predict wether a person will default on a loan.
The Bank Indessa has not done well. Many of their NPA was from loan defaulters. They have collected data
over the years, and can use machine learning models to analyze their data. They want to find the defaulters
using the data.

# About report
The report compares the three algorithm baysian optimization, grid search, and the developed genetic algorithm.

The genetic algorithm showed results comparable to the other algorithms. However, the dataset is not ideal for testing.

Here's an example of a parameter for XGBoost evolving over generations.
It can be seen that the parameter values are spread out at the start, and then converge towards a final parameter value. 

![bilde](https://github.com/user-attachments/assets/240adb3d-8873-46e3-87de-a5f812c96057)

The results and analyzation of the results can be further read in the report, at the sections Results, Discussion and Conclusion.

# About code
The code for the genetic algorithm can be found at ```Assignment_1_DL/evolutionary_tuning.py```.
An overview can be seen at the function ```run()```

The ```requirments.txt``` file holds all dependencies for the project.










