import csv
import math
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import ensemble, neighbors, tree
from sklearn.model_selection import GridSearchCV


def main():
    '''
    Col 1-103: Continuous
    Col 104-106: Binary
    Col 107: Target (Binary)
    '''

    '''
    Read data and test data
    Split into set with target = 0 and target = 1
    '''
    print()
    print(f'READING DATA...')
    path = ''
    # 107x1500
    data = pd.read_csv(path+'Ecoli.csv', header=0)
    # 106x917
    testData = pd.read_csv(path+'Ecoli_test.csv', header=0)
    print(f'DATA READ \n')
    print(f'PRE-PROCESSING DATA...')
    dataTarg0 = data.loc[data['Target (Col 107)'] == 0]
    dataTarg1 = data.loc[data['Target (Col 107)'] == 1]

    '''
    Drop rows with too many NaN
    '''
    perc = 15.0

    df = dataTarg0.copy()
    s = len(df)
    min_count = int(((100-perc)/100)*df.shape[1] + 1)
    df = df.dropna(axis=0, thresh=min_count)
    #print(f"    Removed {s - len(df)} rows")
    dataTarg0 = df.copy()

    df = dataTarg1.copy()
    s = len(df)
    min_count = int(((100-perc)/100)*df.shape[1] + 1)
    df = df.dropna(axis=0, thresh=min_count)
    #print(f"    Removed {s - len(df)} rows")
    dataTarg1 = df.copy()

    dataDroppedNaN = (pd.concat([dataTarg0, dataTarg1], axis=0)).sort_index()
    print(f'    Dropped rows with NaN > {perc}%')
    '''
    Impute and interpolate values
    Continuous: Linear
    Binary: Most frequent
    '''
    # Split the data in binary and continuous
    dataTarg0cont = dataTarg0.iloc[:, :103]
    dataTarg0bin = dataTarg0.iloc[:, 103:106]
    dataTarg1cont = dataTarg1.iloc[:, :103]
    dataTarg1bin = dataTarg1.iloc[:, 103:106]

    # Impute
    imp = SimpleImputer(strategy="most_frequent", missing_values=np.nan)
    dataTarg0bin = pd.DataFrame(imp.fit_transform(
        dataTarg0bin), index=dataTarg0bin.index, columns=dataTarg0bin.columns)
    dataTarg1bin = pd.DataFrame(imp.fit_transform(
        dataTarg1bin), index=dataTarg1bin.index, columns=dataTarg1bin.columns)

    # Interpolate
    dataTarg0cont = dataTarg0cont.interpolate(
        method='linear', limit_direction='both', axis=0)
    dataTarg1cont = dataTarg1cont.interpolate(
        method='linear', limit_direction='both', axis=0)

    # Concatenate
    dataTarg0 = pd.concat([dataTarg0cont, dataTarg0bin,
                          dataTarg0.iloc[:, 106:]], axis=1, join="inner")

    dataTarg1 = pd.concat([dataTarg1cont, dataTarg1bin,
                          dataTarg1.iloc[:, 106:]], axis=1, join="inner")

    dataImputed = (pd.concat([dataTarg0, dataTarg1], axis=0)).sort_index()

    print(f'    Imputed/Interpolated NaN')

    '''
    Outlier Detection (using imputed values)
    '''
    df = dataImputed.iloc[:, :103]
    Q1 = df.quantile(0.05)
    Q3 = df.quantile(0.95)
    IQR = Q3 - Q1

    # dataDropImputeRem means original data, with dropped rows containing too many NaN, imputed NaNs and removed outlier rows
    dataDropImputeRem = dataImputed[~((df < (Q1 - 1.5 * IQR)) |
                                      (df > (Q3 + 1.5 * IQR))).any(axis=1)]

    # dataDropRem means original data, with dropped rows containing too many NaN and removed outlier rows
    dataDropRem = dataDroppedNaN[~((df < (Q1 - 1.5 * IQR)) |
                                   (df > (Q3 + 1.5 * IQR))).any(axis=1)]

    print(f'    Removed outliers')
    '''
    Impute again (with outliers removed)
    '''
    # Take dataDropRem and impute it
    dataTarg0 = dataDropRem.loc[dataDropRem['Target (Col 107)'] == 0]
    dataTarg1 = dataDropRem.loc[dataDropRem['Target (Col 107)'] == 1]
    # Split the data in binary and continuous
    dataTarg0cont = dataTarg0.iloc[:, :103]
    dataTarg0bin = dataTarg0.iloc[:, 103:106]
    dataTarg1cont = dataTarg1.iloc[:, :103]
    dataTarg1bin = dataTarg1.iloc[:, 103:106]

    imp = SimpleImputer(strategy="most_frequent", missing_values=np.nan)
    dataTarg0bin = pd.DataFrame(imp.fit_transform(
        dataTarg0bin), index=dataTarg0bin.index, columns=dataTarg0bin.columns)
    dataTarg1bin = pd.DataFrame(imp.fit_transform(
        dataTarg1bin), index=dataTarg1bin.index, columns=dataTarg1bin.columns)

    # Interpolate
    dataTarg0cont = dataTarg0cont.interpolate(
        method='linear', limit_direction='both', axis=0)
    dataTarg1cont = dataTarg1cont.interpolate(
        method='linear', limit_direction='both', axis=0)

    # Concatenate
    dataTarg0 = pd.concat([dataTarg0cont, dataTarg0bin,
                          dataTarg0.iloc[:, 106:]], axis=1, join="inner")

    dataTarg1 = pd.concat([dataTarg1cont, dataTarg1bin,
                          dataTarg1.iloc[:, 106:]], axis=1, join="inner")

    # Just to collect both availible datasets at one place
    dataDropImputeRem = dataDropImputeRem

    #
    #   dataDropRemImputed:
    #   original data -> drop rows with too many NaNs -> Remove outliers found based on dataDropImpute -> Impute remaining NaNs
    #
    dataDropRemImputed = (
        pd.concat([dataTarg0, dataTarg1], axis=0)).sort_index()
    print(f'    Two sets: dataDropImputeRem, dataDropRemImputed')
    '''
    Normalising
    Using standardisation 
    '''
    print(f'    Normalising')
    df = dataDropRemImputed.copy()
    df2 = df.iloc[:, :103]

    # Min-Max
    scaler = MinMaxScaler()
    df_norm_103 = pd.DataFrame(scaler.fit_transform(
        df2), columns=df2.columns, index=df2.index)
    df_norm = df_norm_103.join(df.iloc[:, 103:])

    df3 = testData.iloc[:, :103]
    df_test_norm_103 = pd.DataFrame(scaler.fit_transform(
        df3), columns=df3.columns, index=df3.index)
    df_test_norm = df_test_norm_103.join(testData.iloc[:, 103:])

    # # Z-score
    # std_scaler = StandardScaler()
    # df_std_103 = pd.DataFrame(
    #     std_scaler.fit_transform(df2), columns=df2.columns, index=df2.index)
    # df_std = df_std_103.join(df.iloc[:, 103:])

    # df_test_std_103 = pd.DataFrame(std_scaler.fit_transform(
    #     df3), columns=df3.columns, index=df3.index)
    # df_test_std = df_test_std_103.join(testData.iloc[:, 103:])

    print(f'DATA AND TEST DATA PROCESSED... \n')
    #
    # We now have the following: (From the data: dataDropRemImputed)
    #   df_norm -> Min-Max data
    #   df_std -> Standardized data
    #   df_test_norm -> Min-Max test data
    #   df_test_std -> Standardized test data
    #

    '''
    TRAIN MODEL
    The input used here should be preprocessed data: Impute -> Outlier detection -> Impute -> Normalize -> DATA TO BE USED

    DATA TO BE USED: df_norm, df_std
    '''

    x_data = df_norm.iloc[:, :106]  # df_std.iloc[:, :106]
    y_data = df_norm.iloc[:, 106:]  # df_std.iloc[:, 106:]
    test_data = df_test_norm  # df_test_std

    def round_calc(res):
        return math.floor(res * 1000)/1000.0

    print(f'TRAINING MODEL...')
    # Decision tree
    '''
    Decision tree
    '''
    print(" Decision tree...")

    # Entropy
    param = {'max_depth': range(1, 20)}
    clf = GridSearchCV(tree.DecisionTreeClassifier(criterion="entropy"),
                       param, cv=30, n_jobs=4, refit="accuracy", scoring=['accuracy', 'f1'])
    clf.fit(X=x_data, y=y_data)
    # clf.predict(test_data)
    accuracy = round_calc(clf.best_score_)
    f1 = round_calc(clf.cv_results_['mean_test_f1'][clf.best_index_])
    accF1 = (accuracy, f1)
    DTE = clf.best_estimator_
    DTE_res = (clf.best_score_)

    print(f'''
    Decision Tree Entropy
    Results: {(accF1)}
    {(clf.best_score_, clf.best_params_)}
        ''')

    # Gini-Index
    param = {'max_depth': range(1, 20)}
    clfg = GridSearchCV(tree.DecisionTreeClassifier(criterion="gini"),
                        param, cv=30, n_jobs=4, refit="accuracy", scoring=['accuracy', 'f1'])
    clfg.fit(X=x_data, y=y_data)
    test_res = list(clfg.predict(test_data))
    accuracy = round_calc(clfg.best_score_)
    f1 = round_calc(clfg.cv_results_['mean_test_f1'][clfg.best_index_])
    accF1 = (accuracy, f1)
    DTG = clfg.best_estimator_
    DTG_res = (clfg.best_score_)

    print(f'''    Decision Tree Gini
    Results: {(accF1)}
    {(clfg.best_score_, clfg.best_params_)}
        ''')

    # Random forest
    '''
    Random forest
    '''
    print(" Random forest...")
    # Entropy
    param = {'max_depth': range(1, 10)}
    clf = GridSearchCV(ensemble.RandomForestClassifier(class_weight="balanced", random_state=42,
                       criterion="entropy"), param, n_jobs=4, refit="accuracy", scoring=['accuracy', 'f1'])
    clf.fit(X=x_data, y=np.ravel(y_data))
    # clf.predict(test_data)
    accuracy = round_calc(clf.best_score_)
    f1 = round_calc(clf.cv_results_['mean_test_f1'][clf.best_index_])
    accF1 = (accuracy, f1)
    RFE = clf.best_estimator_
    RFE_res = clf.best_score_

    print(f'''
    Random Forest Entropy
    Results: {(accF1)}
    {(clf.best_score_, clf.best_params_)}
        ''')

    param = {'max_depth': range(1, 10)}
    clf = GridSearchCV(ensemble.RandomForestClassifier(class_weight="balanced", random_state=42,
                       criterion="gini"), param, n_jobs=4, refit="accuracy", scoring=['accuracy', 'f1'])
    clf.fit(X=x_data, y=np.ravel(y_data))
    # clf.predict(test_data)
    accuracy = round_calc(clf.best_score_)
    f1 = round_calc(clf.cv_results_['mean_test_f1'][clf.best_index_])
    accF1 = (accuracy, f1)
    RFG = clf.best_estimator_
    RFG_res = clf.best_score_

    print(f'''    Random Forest Gini
    Results: {(accF1)}
    {(clf.best_score_, clf.best_params_)}
        ''')

    # Naïve Bayes
    '''
    Naïve Bayes
    '''
    print(" Naïve Bayes...")
    param = {}
    clf = GridSearchCV(GaussianNB(
    ), param, cv=30, n_jobs=4, refit="accuracy", scoring=['accuracy', 'f1'])
    clf.fit(X=x_data, y=np.ravel(y_data))
    # clf.predict(testContents)
    accuracy = round_calc(clf.best_score_)
    f1 = round_calc(clf.cv_results_[
        'mean_test_f1'][clf.best_index_])
    accF1 = (accuracy, f1)
    NB = clf.best_estimator_

    print(f'''
    Naïve Bayes
    Results: {(accF1)}
    {(clf.best_score_, clf.best_params_)}
        ''')

    # k-NN
    '''
    k-NN
    '''
    print(" k-NN...")
    param = {'n_neighbors': np.arange(1, 50)}
    clf = GridSearchCV(neighbors.KNeighborsClassifier(p=2
                                                      ), param, cv=30, n_jobs=4, refit="accuracy", scoring=['accuracy', 'f1'])
    clf.fit(X=x_data, y=np.ravel(y_data))
    # clf.predict(testContents)
    accuracy = round_calc(clf.best_score_)
    f1 = round_calc(clf.cv_results_['mean_test_f1'][clf.best_index_])
    accF1 = (accuracy, f1)
    KNN = clf.best_estimator_

    print(f'''
    k-NN
    Results: {(accF1)}
    {(clf.best_score_, clf.best_params_)}
        ''')

    # Ensemble: 'Decision Tree Gini','Decision Tree Entropy', 'k-NN', 'Random Forest Entropy', 'Random Forest Gini'
    '''
    Ensemble
    '''
    print(" Ensemble...")
    classifiers = [('Decision Tree Gini', DTG), ('Decision Tree Entropy', DTE), ('k-NN', KNN),
                   ('Random Forest Entropy', RFE), ('Random Forest Gini', RFG)]
    param = {}
    clf = GridSearchCV(ensemble.VotingClassifier(
        classifiers), param, cv=30, n_jobs=4, refit="accuracy", scoring=['accuracy', 'f1'])
    clf.fit(X=x_data, y=np.ravel(y_data))
    accuracy = round_calc(clf.best_score_)
    f1 = round_calc(clf.cv_results_['mean_test_f1'][clf.best_index_])
    accF1 = (accuracy, f1)
    test_res = list(clf.predict(test_data))
    ENS = clf.best_estimator_

    print(f'''
    Ensemble (DTG, DTE, KNN, RFE, RFG)
    Results: {(accF1)}
    {(clf.best_score_, clf.best_params_)}
        ''')


if __name__ == "__main__":
    main()
