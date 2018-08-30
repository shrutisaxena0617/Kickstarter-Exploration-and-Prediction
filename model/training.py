import pandas as pd
import numpy as np

# # Visualization
# from matplotlib import pyplot as plt
# import seaborn as sns
# %matplotlib inline

# Classifiers
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Model Selection and Evaluation
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Other
from datetime import datetime
import dill as pickle


class KickstarterModel:

    def __init__(self):
        """Initialize cols list to specify model features

        Filename saves the pickled model in the format model_v<version_number>.pk
        """

        self.cols = ['ID', 'is_successful', 'backers', 'usd_pledged_real', 'usd_goal_real', 'duration_running', 'is_US',
                     'main_category_Art', 'main_category_Comics', 'main_category_Crafts', 'main_category_Dance',
                     'main_category_Design', 'main_category_Fashion', 'main_category_Film & Video', 'main_category_Food',
                     'main_category_Games', 'main_category_Journalism', 'main_category_Music', 'main_category_Photography',
                     'main_category_Publishing', 'main_category_Technology', 'main_category_Theater']
        self.filename = 'model_v1.pk'

    def preprocess(self, raw_data):
        """Preprocessing training data to handle null values, build new features, and encode categorical data

        data_final is the final training dataframe object
        """

        raw_data['name'].fillna('Unnamed', inplace=True)
        raw_data.drop(['usd pledged', 'currency', 'goal', 'pledged'], axis=1, inplace=True)
        raw_data['deadline'] = pd.to_datetime(raw_data['deadline'], infer_datetime_format=True)
        raw_data['launched'] = pd.to_datetime(raw_data['launched'], infer_datetime_format=True)
        raw_data.drop(raw_data[raw_data.launched.dt.year == 1970].index, inplace=True)
        raw_data['duration_running'] = (raw_data['deadline'] - raw_data['launched']).dt.days
        raw_data.drop(['deadline', 'launched'], axis=1, inplace=True)
        raw_data.drop(raw_data[(raw_data.state != 'successful') & (raw_data.state != 'failed')].index, inplace=True)
        raw_data.drop(raw_data[raw_data.country == 'N,0"'].index, inplace=True)
        raw_data['is_successful'] = (raw_data['state'] == 'successful').astype(int)
        raw_data['is_US'] = (raw_data['country'] == 'US').astype(int)

        data = pd.get_dummies(raw_data, columns=["main_category"])
        data.drop(['category', 'country', 'state', 'name'], axis=1, inplace=True)

        data_final = data[self.cols]

        return data_final

    def split_for_validation(self, data_final):
        """Split training data in a 80-20 split ratio"""

        X = data_final.drop(['ID', 'is_successful'], axis=1)
        y = data_final['is_successful']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        return X_train, X_test, y_train, y_test

    def train(self, X_train, X_test, y_train, y_test):
        """Train the model with decision tree classifier

        Apply grid search for hyperparamter tuning

        Display model accuracy on test dataset
        """

        clf = DecisionTreeClassifier()
        model = clf.fit(X_train, y_train)
        pred = model.predict(X_test)
        accuracy = accuracy_score(pred, y_test)

        param_range = range(1, 20)
        param_grid = {'max_depth': param_range}
        gs = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=-1)
        gs = gs.fit(X_train, y_train)

        gs_model = gs.best_estimator_
        final_model = gs_model.fit(X_train, y_train)

        print('Test accuracy: %.3f' % final_model.score(X_test, y_test))

        return final_model


def main():
    """Train the model on loaded csv data

    Serialize the trained model
    """

    raw_data = pd.read_csv('../data/data.csv', sep=',')
    kickstarter_model = KickstarterModel()
    data_final = kickstarter_model.preprocess(raw_data)
    X_train, X_test, y_train, y_test = kickstarter_model.split_for_validation(data_final)
    final_model = kickstarter_model.train(X_train, X_test, y_train, y_test)
    with open('../api/models/' + kickstarter_model.filename, 'wb') as file:
        pickle.dump(final_model, file)


if __name__ == '__main__':
    main()
