



import pandas as pd


class Dataset:
    def __init__(self):
        self.folder_name = "dataset/"
        self.data_name_train = "train_indessa.csv"
        self.data_name_test = "test_indessa.csv"
        self.filename_train = self.folder_name + self.data_name_train
        self.filename_test = self.folder_name + self.data_name_test

        self.target = 'loan_status'

        self.df_train = pd.read_csv(self.filename_train)
        self.df_test = pd.read_csv(self.filename_test)

    def get_dataset(self):
        # Separate target from training and testing data
        y_train = self.df_train[self.target]
        x_train = self.df_train.drop(columns=[self.target])

        y_test = self.df_test[self.target] # TODO: fail
        x_test = self.df_test.drop(columns=[self.target])

        return x_train, y_train, x_test, y_test

    def alt_get_dataset(self):
        y = self.df_train[self.target]
        x = self.df_train.drop(columns=[self.target])
        return x, y


    def get_dataset_train(self):
        return self.df_train




