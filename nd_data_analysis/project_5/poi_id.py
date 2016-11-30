#!/usr/bin/python

import sys
import pickle
import pandas as pd
import numpy as np
import math
sys.path.append("../tools/")

### dataset related imports
from sklearn.cross_validation import train_test_split
from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import cross_validation
from sklearn.decomposition import PCA
from sklearn.decomposition import RandomizedPCA
### classifier imports
##############################################################
######## Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
######## Decision Trees
from sklearn.tree import DecisionTreeClassifier
######## Support Vector Machines
from sklearn.svm import SVC
######## Regression
from sklearn.linear_model import LogisticRegression
######## Others
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

###############################################################################
### GLOBAL VARIABLES
###############################################################################

# output files
output_file_results = "results.csv"

# show results of SelectKBest
show_SelectKBest_results = False

# global flags to control basic functionality
create_new_message_features = False
create_new_finance_features = False

# wanna perform PCA for feature selection
do_perform_PCA = False

### define number of components available after PCA
pca_components = 7

# train_test_split configuration
gl_test_size    = 0.2 # array to provide different test sizes to check

### featureFormat configuration
gl_sort_keys    =   True

# random sate config
gl_random_state = None

###############################################################################
### list of all classifiers and related configurations
gl_clf_list = [
    #[1, GaussianNB,
    #    {},
    #    "scaled"],
    #[2, SVC,
    #    {'kernel': ['linear']}, # - 'kernel': ('linear')'rbf'
    #     "scaled"],
    #[3, LogisticRegression,
    #    {"C": [1, 10, 1000, 0.1, 0.5], # - 10, 10 ** 2, 10000, 1, 0.1, 0.2, 0.5, 100, 1000
    #     "tol": [0.5, 10 ** -10, 10 ** -5], # - , 10 ** -10, 10 ** -5
    #     #"multi_class": ('ovr', 'multinomial')},
    #     "solver": ('sag', 'lbfgs')}, # 'newton-cg', 'lbfgs', 'liblinear', 'sag'
    #     "scaled"],
    #[4, KNeighborsClassifier, # almost there
    #    {"n_neighbors":[3], # - 2, 3, 4, 5, 7, 10 # increasing the number increases the precision
    #     "p":[3], # 2, 3, 4
    #     "algorithm": ['auto'],
    #     "leaf_size": [5],
    #     "weights": ['uniform']},
    #     "scaled"],
    #[5, DecisionTreeClassifier,
    #    {"min_samples_split":[2, 3, 4, 5, 7]}, # - 4, 5, 7,
    #     "not scaled"],
    [6, AdaBoostClassifier,
        {"n_estimators":[3], # 3, 5, 10, 30, 40, 50, 60, 70],
         "algorithm": ['SAMME'],
         "learning_rate": [1.5]}, # - 'SAMME', 'SAMME.R'
         "scaled"],
    #[7, RandomForestClassifier,
    #    {"n_estimators":[3, 5, 4, 7, 10, 15, 20], # - 3, 5, 4, 7, 10, 15, 20
    #     "criterion": ('gini', 'entropy')},
    #     "not scaled"]
    ]

###############################################################################
### clf info collection about performance of classifiers with given arguments
### (using a pandas dataframe for getting stats and cals easily done)
clf_collection = pd.DataFrame(columns=['class_id', 'clf', 'kwargs',
                                       'features_scaled',
                                       'test set size in \%',
                                       'number of features', 'features',
                                       'accuracy', 'precision', 'recall',
                                       'best parameters', 'best estimator'])

clf_best_collection = pd.DataFrame(columns=['class_id', 'clf', 'features_scaled',
                                            'number of features', 'features',
                                            'accuracy','precision', 'recall',
                                            'best parameters', 'best estimator',
                                            'orig data set',
                                            'create_new_message_features',
                                            'create_new_finance_features',
                                            'PCA'])

###############################################################################
###############################################################################
### global definition of main originally available features
### Feature list from the email data
features_email = [
    #"from_messages",
    # 'email_address', -> not needed for analysis
    "from_poi_to_this_person",
    #"from_this_person_to_poi",
    #"shared_receipt_with_poi",
    #"to_messages"
    ]

### Feature list from the financial data
features_financial = [
    "bonus",
    #"salary",
    #"deferral_payments",
    #"deferred_income",
    #"director_fees",
    #"exercised_stock_options",
    "expenses",
    #"loan_advances",
    #"long_term_incentive",
    #"other",
    #"restricted_stock",
    #"restricted_stock_deferred",
    "total_payments",
    #"total_stock_value"
    ]

# -----------------------------------------------------------------------------
# Name: getFeatureList
# -----------------------------------------------------------------------------
# Description:  function which is composing the different features in a list
#               before it returns the complete list.
# -----------------------------------------------------------------------------
# Output:   feature_list       - data set including all original features cleanup
#                             in respect to outliers
# -----------------------------------------------------------------------------
def getFeaturesList():
    ### The first feature must be "poi".
    poi = ['poi']
    return poi + features_email + features_financial

# -----------------------------------------------------------------------------
# Name: clean_data
# -----------------------------------------------------------------------------
# Description:  function removing outliers from the data set. for detecting the
#               outliers, a manual qualitative approach has been chosen. Means
#               as a result of reviewing the data set in detail, some records
#               inside the data set which need to be removed as they are
#               distorting the data have been identified and will be removed.
# -----------------------------------------------------------------------------
# Input:    data_dict       - data set including all original features
# -----------------------------------------------------------------------------
# Output:   data_dict       - data set including all original features cleanup
#                             in respect to outliers
# -----------------------------------------------------------------------------
def clean_data(data_dict):

    # remove the outliers from the data set
    data_dict.pop("TOTAL", 0)
    data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)

    return data_dict

# -----------------------------------------------------------------------------
# Name: create_new_features
# -----------------------------------------------------------------------------
# Description:  function creating new features based on the original features
#               and adding those new features to the data set
# -----------------------------------------------------------------------------
# Input:    data_dict       - data set including all original features already
#                             cleanup in respect to outliers
# -----------------------------------------------------------------------------
# Output:   data_dict       - data set including all original features already
#                             cleanup in respect to outliers enriched with
#                             additional features created upon original
#                             features
# -----------------------------------------------------------------------------
def create_new_features(data_dict, my_feature_list):

    if create_new_message_features:
        #######################################################################
        # new message feature list
        new_message_features = [
                                 "total_messages"
                                ,"total_poi_related_messages"
                                ,"total_poi_message_ratio"
                                ]

    if create_new_finance_features:
        #######################################################################
        # new financial feature list - is getting filled when creating features
        new_financial_features = []

        # get all new financial feature names set up
        for feature in features_financial:
            name = feature + "_squared"
            new_financial_features.append(name)
            name = feature + "_log"
            new_financial_features.append(name)

    #######################################################################
    # create features
    for rec in data_dict:

        if  create_new_message_features:
            #######################################################################
            # new features related to message data

            # total message of a person
            data_dict[rec]["total_messages"] = \
                isValNumber(data_dict[rec]["from_messages"]) + \
                isValNumber(data_dict[rec]["to_messages"])

            # total messages of a person somehow related to a POI
            data_dict[rec]["total_poi_related_messages"] = \
                isValNumber(data_dict[rec]["from_poi_to_this_person"]) + \
                isValNumber(data_dict[rec]["from_this_person_to_poi"]) + \
                isValNumber(data_dict[rec]["shared_receipt_with_poi"])

            # ratio between total message and total POI related messages count
            try:
                data_dict[rec]["total_poi_message_ratio"] = float( \
                    isValNumber(data_dict[rec]["total_poi_related_messages"]) / \
                    isValNumber(data_dict[rec]["total_messages"]))
            except ZeroDivisionError:
                data_dict[rec]["total_poi_message_ratio"] = 0


        if create_new_finance_features:
            #######################################################################
            # new features related to financial data

            # square existing financial features
            for feature in features_financial:
                name = feature + "_squared"
                try:
                    number = isValNumber(data_dict[rec][feature])
                    data_dict[rec][name] = number ** 2
                except:
                    data_dict[rec][name] = "NaN"

            # square existing financial features
            for feature in features_financial:
                name = feature + "_log"
                try:
                    number = isValNumber(data_dict[rec][feature])
                    data_dict[rec][name] = math.log(number)
                except:
                    data_dict[rec][name] = "NaN"

    #######################################################################
    # composing complete list

    if create_new_finance_features and create_new_message_features:
        my_feature_list = my_feature_list + new_message_features \
                          + new_financial_features
        return data_dict, my_feature_list

    if create_new_finance_features and not create_new_message_features:
        my_feature_list = my_feature_list + new_financial_features
        return data_dict, my_feature_list

    if not create_new_finance_features and create_new_message_features:
        my_feature_list = my_feature_list + new_message_features
        return data_dict, my_feature_list

    if not create_new_finance_features and not create_new_message_features:
        return data_dict, my_feature_list
# -----------------------------------------------------------------------------
# Name: isValNumber
# -----------------------------------------------------------------------------
# Description:  checks if a values is a valid number
# -----------------------------------------------------------------------------
# Input:    number       -  input values to be checked for being number
# -----------------------------------------------------------------------------
# Output:   number       -  number in float format
# -----------------------------------------------------------------------------
def isValNumber(number):
    if number == "NaN":
        return 0
    else:
        return float(number)



###############################################################################
### CLASSIFIERS
###############################################################################
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
###############################################################################

# -----------------------------------------------------------------------------
# Name: do_PCA
# -----------------------------------------------------------------------------
# Description:  function which performes a PCA (principal component analysis)
#               on the training and test features to select/transform the
#               features to increase simplicity without impacting the
#               significancy of the data.
# -----------------------------------------------------------------------------
# Input:    features_train  - features training data set portion
#           features_test   - features test data set portion
# -----------------------------------------------------------------------------
# Output:   X_train_pca     - PCA transformed training data set portion
#           X_test_pca      - PCA transformed test data set portion
# -----------------------------------------------------------------------------
def do_PCA(features_train, features_test, n_pca_comps):

    if do_perform_PCA:
        # initialize PCA
        # pca = RandomizedPCA(n_components=pca_components, whiten=True).fit(features_train)
        pca = PCA(n_components=n_pca_comps, whiten=True).fit(features_train)

        # apply PCA
        X_train_pca = pca.transform(features_train)
        X_test_pca = pca.transform(features_test)

        print "doing PCA with " + str(n_pca_comps) + " target components"

        return X_train_pca, X_test_pca

    else:
        print "no PCA applied"
        return features_train, features_test

# -----------------------------------------------------------------------------
# Name: apply_clfs
# -----------------------------------------------------------------------------
# Description:  function which organizes the data for testing and trainings
#               purposes using the features list which provides information
#               about the features which are going to be used for testing and
#               training.
#               Further the function is picking up the global definitions of
#               all classifiers, iterates through the list and triggers
#               necessary functions to train and predict the algorithms and
#               further returns back the scores related to the performance of
#               the algorithms.
# -----------------------------------------------------------------------------
# Input:    features_train          - features of train data set
#           features_test           - features of test data set
#           features_train_scaled   - scaled features of train data set
#           features_test_scaled    - scaled features of test data set
#           labels_train            - labels of train data set
#           labels_test             - labels of test data set
#           size                    -
#           my_feature_list         -
# -----------------------------------------------------------------------------
# Output:   clf_collection          - collection of best performing
#                                     classification algorithm
# -----------------------------------------------------------------------------
def apply_clfs(features_train, features_test,
               features_train_scaled, features_test_scaled,
               labels_train, labels_test, size, my_feature_list):

    # get idx for adding to the global clf_collection data frame
    idx = len(clf_collection.index)

    # iterate through list of defined classifiers
    for id, clf_class, clf_kwargs, scale_info in gl_clf_list:
        #print "#############################################################"
        #print "now checking: " + str(clf_class)
        #print "best configuration for algorithm: "

        if scale_info == "scaled":
            # get scores of specific classifier
            precision, recall, accuracy, best_params, best_estimator = \
                get_classifier_scores(clf_class, clf_kwargs, features_train,
                                    features_test, labels_train, labels_test,
                                      my_feature_list)

            print "Training: " + str(clf_class) + \
                  ", # feature scaling: TRUE"

        else:
            # get scores of specific classifier
            precision, recall, accuracy, best_params, best_estimator = \
                get_classifier_scores(clf_class, clf_kwargs, features_train_scaled,
                                      features_test_scaled, labels_train, labels_test,
                                      my_feature_list)

            print "Training: " + str(clf_class) + \
                  ", # feature scaling: FALSE (not considering PCA)"

        # add results to the clf_collection data frame
        clf_collection.loc[idx] = [id, clf_class, clf_kwargs, scale_info,
                                   size * 100, len(my_feature_list),
                                   str(my_feature_list), accuracy, precision,
                                   recall, best_params, best_estimator]

        #print clf_collection.loc[idx]
        idx += 1


    # figure out best algorithm
    # and return the wisster
    return clf_collection

###############################################################################
### TUNE CLASSIFIERS
###############################################################################
### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.
###                                                 StratifiedShuffleSplit.html
###############################################################################

# -----------------------------------------------------------------------------
# Name: get_classifier_scores
# -----------------------------------------------------------------------------
# Description:  function which creates and trains a classifier. The created
#               classifier creates prediction based on the test features. Based
#               on the result the classifier gets scored from a accuracy as
#               well as from a confusion matrix point of view.
# -----------------------------------------------------------------------------
# Input:    clf_class       - classifier name
#           clf_kwargs      - key word args/params to create classifier
#           features_train  - features of training data set portion
#           features_test   - features of test data set portion
#           labels_train    - labels of training data set portion
#           labels_test     - labels of test data set portion
# -----------------------------------------------------------------------------
# Output:   scores          - dictionary which key performance scores
#            - accuracy     - accuracy of predictions
#            - precision    - of all positive labeled items, how many truly
#                             belong to the positive class
#            - recall       - how many positive items where recalled
# -----------------------------------------------------------------------------
def get_classifier_scores(clf_class, clf_kwargs, features_train, features_test,
                          labels_train, labels_test, feature_list):

    # instantiate classifier with related arguments
    clf = clf_class()

    # set up cross validation
    crossval = cross_validation.StratifiedShuffleSplit(
                                labels_train,
                                50,
                                test_size=gl_test_size,
                                random_state=gl_random_state)

    # perform grid search to find optimal parameter configuration
    grid_search = GridSearchCV(clf, clf_kwargs, cv=crossval, scoring='recall')

    #grid_search = GridSearchCV(clf, clf_kwargs, scoring='recall')

    # train
    grid_search.fit(features_train, labels_train)

    # pick a winner
    best_clf = grid_search.best_estimator_

    # predict for test features
    predictions = best_clf.predict(features_test)

    # calculate accuracy
    scores = dict()
    scores["accuracy"] = accuracy_score(labels_test, predictions)
    scores["precision"] = precision_score(labels_test, predictions)
    scores["recall"] = recall_score(labels_test, predictions)

    # declare as string else you point to a changing value
    best_configuration = ""
    best_configuration = str(grid_search.best_estimator_)

    # Print the feature ranking
    try:
        # Get importance of features
        importances = best_clf.feature_importances_
        indices = np.argsort(importances)[::-1]

        print("Feature ranking:")
        for f in range(features_train.shape[1]):
            print("%d. feature %s (%f)" % (f+1, feature_list[indices[f]], importances[indices[f]]))

    except:
        print "no importances available for classifier " + str(clf_class)

    # return cm scores and accuracy
    return scores["precision"], scores["recall"], scores["accuracy"], \
           grid_search.best_params_, best_configuration

###############################################################################
### START MAIN FUNCTION
###############################################################################
def main():

    ###########################################################################
    ### Get features from global definitions
    orig_features_list = getFeaturesList()

    ### load the dictionary containing the dataset
    with open("final_project_dataset.pkl", "r") as data_file:
        data_dict = pickle.load(data_file)

    ### Store to my_dataset for easy export below.
    orig_dataset = data_dict

    ###########################################################################
    ### Remove outliers
    orig_dataset = clean_data(orig_dataset)

    ###########################################################################
    ### Create new feature(s)
    my_features_list = orig_features_list
    my_dataset, my_features_list = create_new_features(orig_dataset, my_features_list)

    ### write full data to file
    data_df = pd.DataFrame(my_dataset)
    data_df.T.to_csv("full_data.csv", sep=',', encoding='utf-8')

    ###########################################################################
    ### Extract features and labels from dataset for local testing

    my_data = featureFormat(my_dataset, my_features_list, sort_keys=gl_sort_keys)
    my_labels, my_features = targetFeatureSplit(my_data)

    ### orig data
    orig_data = featureFormat(orig_dataset, orig_features_list, sort_keys=gl_sort_keys)
    orig_labels, orig_features = targetFeatureSplit(orig_data)

    ###########################################################################
    ### Preparation of training and testing data

    # without feature scaling
    my_features_train, my_features_test, my_labels_train, my_labels_test = \
        train_test_split(my_features, my_labels, test_size=gl_test_size,
                         random_state=gl_random_state)

    orig_features_train, orig_features_test, orig_labels_train, orig_labels_test = \
        train_test_split(orig_features, orig_labels, test_size=gl_test_size,
                         random_state=gl_random_state)

    if show_SelectKBest_results:
        print "BEST 10 FEATURES"
        bestTen = SelectKBest(f_classif, k=5)
        bestTen.fit(my_features_train, my_labels_train)

        try:
            scores = bestTen.scores_
            indices = np.argsort(scores)[::-1]
            print("Features score ranking based on SelectKBest:")
            for f in range(np.array(my_features_train).shape[1]):
                print("%d. feature %s (%f)" % (f + 1, my_features_list[indices[f]], scores[indices[f]]))
        except:
            print "no scores available for the given combination"

    ### with feature scaling
    scaler = MinMaxScaler()
    my_features_scaled = scaler.fit_transform(my_features)
    my_features_train_scaled, my_features_test_scaled, \
    my_labels_train_scaled, my_labels_test_scaled = \
        train_test_split(my_features_scaled, my_labels, test_size=gl_test_size,
                         random_state=gl_random_state)

    orig_features_scaled = scaler.fit_transform(orig_features)
    orig_features_train_scaled, orig_features_test_scaled, \
    orig_labels_train_scaled, orig_labels_test_scaled = \
        train_test_split(orig_features_scaled, orig_labels, test_size=gl_test_size,
                         random_state=gl_random_state)

    print "Current test  data size:  " + str(gl_test_size * 100) + " %"
    print "Current train data size:  " + str(100 - gl_test_size * 100) + " %"

    ###########################################################################
    ### PCA

    # prepare info for later output
    PCA_info = do_perform_PCA
    if PCA_info:
        PCA_info = pca_components

    # Do a PCA on the features for non scaled data
    my_features_train, my_features_test = \
        do_PCA(my_features_train, my_features_test, pca_components)

    # Do a PCA on the features for  scaled data
    my_features_train_scaled, my_features_test_scaled = \
        do_PCA(my_features_train_scaled, my_features_test_scaled, pca_components)

    # Do a PCA on the features for non scaled data
    #orig_features_train, orig_features_test = \
    #    do_PCA(orig_features_train, orig_features_test, pca_components)

    # Do a PCA on the features for  scaled data
    #orig_features_train_scaled, orig_features_test_scaled = \
    #    do_PCA(orig_features_train_scaled, orig_features_test_scaled, pca_components)

    ###########################################################################
    ### Train Classifier(s)

    print "###################################################################"
    print "Start performing selection of best algorithms and configurations "

    # calling the classifier validation with non-scaled features
    apply_clfs(my_features_train, my_features_test,
               my_features_train_scaled, my_features_test_scaled,
               my_labels_train, my_labels_test, gl_test_size, my_features_list)

    print "End performing selection of best algorithms and configurations "
    print "###################################################################"


    # pick 10 best performing classifier
    best_clf_config_list = clf_collection.sort_values(['precision','recall',
                                            'accuracy','number of features'],
                                    ascending=[False,False,False,True])

    clf_collection.sort_values(['precision', 'recall',
                                'accuracy', 'number of features'],
                               ascending=[False, False, False, True])

    # dump the results of all the tested classifiers and related configurtion
    # and train/test setup
    clf_collection.to_csv("training_data.csv", sep=',', encoding='utf-8')

    # iterating through all the classifiers chosen
    print "Validating list of best classifiers: "
    for index, best_clf_config in best_clf_config_list.iterrows():
        # go for the best, instantiate it and dump the data
        best_clf_class_id = best_clf_config["class_id"]
        best_clf_params = best_clf_config["best parameters"]

        for id, clf_class, clf_kwargs, feat_scaling in gl_clf_list:
            if id == int(best_clf_class_id):

                try:
                    # instantiate classifier
                    best_clf = clf_class(**best_clf_params)
                    best_clf_org = clf_class(**best_clf_params)

                    if best_clf_config["features_scaled"]:
                        # train the algorithm
                        best_clf.fit(my_features_train_scaled, my_labels_train)
                        best_clf_org.fit(orig_features_train_scaled, orig_labels_train)
                    else:
                        # train the algorithm
                        best_clf_org.fit(orig_features_train, orig_labels_train)
                        best_clf.fit(my_features_train, my_labels_train)

                    print "start original data set"
                    # test with original data set
                    #v_o_total_predictions, v_o_accuracy, v_o_precision, v_o_recall, v_o_f1, v_o_f2 =\
                    #    test_classifier(best_clf_org, orig_dataset, orig_features_list)

                    #clf_best_collection.loc[1000 + index] = (best_clf_config["class_id"],
                    #                                  best_clf_config["clf"],
                    #                                  best_clf_config['features_scaled'],
                    #                                  len(orig_features_list),
                    #                                  str(my_features_list),
                    #                                  v_o_accuracy, v_o_precision,
                    #                                  v_o_recall, best_clf_params,
                    #                                  best_clf_config["best estimator"],
                    #                                  True,
                    #                                  create_new_message_features,
                    #                                  create_new_finance_features,
                    #                                  PCA_info)

                    # dump final information
                    dump_classifier_and_data(best_clf, my_dataset, my_features_list)

                    print "start original data set with new features"
                     #test with newly created features on top of the original data set
                    v_total_predictions, v_accuracy, v_precision, v_recall, v_f1, v_f2 =\
                        test_classifier(best_clf, my_dataset, my_features_list, do_perform_PCA,
                            pca_components, best_clf_config['features_scaled'])

                    clf_best_collection.loc[index] = (best_clf_config["class_id"],
                                                      best_clf_config["clf"],
                                                      best_clf_config['features_scaled'],
                                                      len(my_features_list),
                                                      str(my_features_list),
                                                      v_accuracy, v_precision, v_recall,
                                                      best_clf_params,
                                                      best_clf_config["best estimator"],
                                                      False,
                                                      create_new_message_features,
                                                      create_new_finance_features,
                                                      PCA_info
                                                      )
                except TypeError:
                    clf_best_collection.loc[index] = (best_clf_config["class_id"],
                                                      best_clf_config["clf"],
                                                      best_clf_config['features_scaled'],
                                                      len(my_features_list),
                                                      str(my_features_list),
                                                      "Error", "Error", "Error",
                                                      best_clf_params,
                                                      best_clf_config["best estimator"],
                                                      False,
                                                      create_new_message_features,
                                                      create_new_finance_features,
                                                      PCA_info
                                                     )

    cbc = clf_best_collection.sort_values(['precision', 'recall', 'accuracy'],
                                                 ascending=[False, False,
                                                            False])

    # write classifier validation result to file
    cbc.to_csv(output_file_results, sep=',', encoding='utf-8')

    print "###################################################################"
    print "###################################################################"

###############################################################################
### END MAIN FUNCTION
###############################################################################

### call main() function
if __name__ == '__main__':
    main()