import pandas as pd;
import numpy as np;

#from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge;
from sklearn.feature_selection import VarianceThreshold;
from sklearn.feature_selection import chi2;

from sklearn.linear_model import Lasso;


'''
All the util functions
'''
def zap_zerovar_cols(data):
    """
    Given a dataframe remove the zero variance columns
    """
    std = data.std();
    zerovar_cols = std[std == 0.0].index;
    return data.drop(zerovar_cols, axis=1), zerovar_cols

def fit_model(data, labels, test_data):
    # now split the data into train and test
    msk = np.random.rand(len(data)) < 0.8;
    train = data[msk];
    y_train = labels[msk];
    test = data[~msk];
    y_test = labels[~msk];

    # fit a linear regression model
    #model = LinearRegression(fit_intercept=True, normalize=True);
    #model = LinearRegression(normalize=True);
    l2 = np.arange(10, 1000, 100)
    scores = {};
    for reg in l2:
        model = Ridge(alpha = reg );
        model = model.fit(train, y_train);
        score = model.score(test, y_test);
        test_predictions = model.predict(test_data);

        scores[str(reg)]= score;

    for val in scores.keys():
        print ">>>>>>> ", val, scores[val];


def fit_lasso(data, labels):
    msk = np.random.rand(len(data)) < 0.8;
    train = data[msk];
    y_train = labels[msk];
    test = data[~msk];
    y_test = labels[~msk];
    lasso_models = {};
    for alpha in np.arange(0.0001, 0.05 ,0.005 ):
        lasso_sel = Lasso(alpha = alpha, tol=0.001);
        lasso_sel.fit(train,y_train);
        coefs = lasso_sel.coef_;
        print "Alpha : ", alpha, " Number of non zero coeffs : ", len( coefs [ coefs != 0] ), " Score on test : ", lasso_sel.score(test, y_test);
        lasso_models[str(alpha)] = lasso_sel;
    return lasso_models;
   

# read the data
data = pd.read_csv('train.csv');
test_data = pd.read_csv('test.csv');
labels = data.loc[:, 'y'];
data = data.drop('y', axis = 1);



# Extract numeric data
numeric_columns = data._get_numeric_data().columns;
numeric_data = data.loc[:, numeric_columns];
numeric_data_test = test_data.loc[:, numeric_columns];

#Convert categorical colummns to one hot encoded columns
cat_cols = set(data.columns) - set(numeric_columns);
cat_data = pd.get_dummies(data, columns = list(cat_cols));
cat_data_test = pd.get_dummies(test_data, columns=list(cat_cols));


data = data.drop(list(cat_cols), axis=1); #drop the categorical columns
test_data = test_data.drop(list(cat_cols), axis=1);

# Zap the zero variance columns
data, zerovar_cols = zap_zerovar_cols(data);
test_data = test_data.drop(zerovar_cols, axis=1);
print "Num of cols in data : ", len(data.columns)
print "Num of cols in cat_data : ", len(cat_data.columns)
data= pd.concat([data, cat_data], axis=1);
test_data = pd.concat([test_data, cat_data_test], axis=1);


lass_models = fit_lasso(data, labels);


