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

def fit_model(data, labels):
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
    models = {};
    for reg in l2:
        model = Ridge(alpha = reg );
        model = model.fit(train, y_train);
        score = model.score(test, y_test);
        models[score]= model;

    for val in models: 
        print ">>>>>>> ", val, models[val];
    min_test= min(models)
    return models[min_test]


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
        test_score = lasso_sel.score(test, y_test);
        print "Alpha : ", alpha, " Number of non zero coeffs : ", len( coefs [ coefs != 0] ), " Score on test : ",test_score; 
        lasso_models[test_score] = lasso_sel;
    min_error = min(lasso_models )
    return lasso_models[min_error];
   


"""
Script starts from here
"""

# read the data
data = pd.read_csv('train.csv');
test_data = pd.read_csv('test.csv');
labels = data.loc[:, 'y'];
print "22222222222222222222222222222222222222222222222"
print "Shape of the training data : ", data.shape
print "Shape of the test data : ", test_data.shape


"""
Preprocessing
"""
data = data.drop(['y', 'ID'], axis = 1);
# Extract numeric data
numeric_columns = data._get_numeric_data().columns;
numeric_data = data.loc[:, numeric_columns];
numeric_data_test = test_data.loc[:, numeric_columns];

#Convert categorical colummns to one hot encoded columns
cat_cols = set(data.columns) - set(numeric_columns);
cat_data = pd.get_dummies(data, columns = list(cat_cols));
cat_data_test = pd.get_dummies(test_data, columns=list(cat_cols));

# drop the categorical columns
data = data.drop(list(cat_cols), axis=1);
test_data = test_data.drop(list(cat_cols), axis=1);



print "33333333333333333333333333333333333333333333333"
print "Shape of training data : ", data.shape
print "Shape of the test data : ", test_data.shape

# Zap the zero variance columns
data, zerovar_cols = zap_zerovar_cols(data);
test_data = test_data.drop(zerovar_cols, axis=1);


print "44444444444444444444444444444444444444444444444444"
print "Shape of training data : ", data.shape
print "Shape of the test data : ", test_data.shape





print "Num of cols in data : ", len(data.columns)
print "Num of cols in cat_data : ", len(cat_data.columns)
data= pd.concat([data, cat_data], axis=1);
test_data = pd.concat([test_data, cat_data_test], axis=1);

print "111111111111111111111111111111111111111111111111111"
print "Shape of the training data : ", data.shape
print "Shape of the test data : ", test_data.shape



lasso_model = fit_lasso(data, labels);
data = data.drop(lasso_model.coef_==0, axis=1);
test_data = test_data.drop(lasso_model.coef_ == 0, axis =1);

print "Shape of the training data : ", data.shape
print "Shape of the test data : ", test_data.shape

final_model = fit_model(data, labels);
test_ids = test_data.loc[:, 'ID'];
test_data = test_data.drop('ID', axis = 1);
test_predictions = final_model.predict(test_data);

output = pd.concat(test_ids, test_predictions, axis = 1);

output.to_csv('submission.csv');

