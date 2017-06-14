import pandas as pd;
import numpy as np;

#from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge;
from sklearn.feature_selection import VarianceThreshold;

'''
All the util functions
'''
def zap_zerovar_cols(data):
    """
    Given a dataframe remove the zero variance columns
    """
    std = data.std();
    zerovar_cols = std[std == 0.0].index;
    return data.drop(zerovar_cols, axis=1)
# read the data
data = pd.read_csv('train.csv');


# Extract numeric data
numeric_columns = data._get_numeric_data().columns;
numeric_data = data.loc[:, numeric_columns];


#Convert categorical colummns to one hot encoded columns
cat_cols = set(data.columns) - set(numeric_columns);
cat_data = pd.get_dummies(data, columns = list(cat_cols));

data = data.drop(list(cat_cols), axis=1); #drop the categorical columns

data = zap_zerovar_cols(data);
data= pd.concat([data, cat_data], axis=1);


# now split the data into train and test
msk = np.random.rand(len(data)) < 0.8;
train = data[msk];
test = data[~msk];

# remove the label or y from data
y_train = train.loc[:,'y']
train = train.drop('y', axis=1);

y_test = test.loc[:, 'y'];
test = test.drop('y', axis=1);


# fit a linear regression model
#model = LinearRegression(fit_intercept=True, normalize=True);
#model = LinearRegression(normalize=True);
l2 = np.arange(10, 1000, 100)
scores = {};
for reg in l2:
    model = Ridge(alpha = reg );
    model = model.fit(train, y_train);
    score = model.score(test, y_test);
    scores[str(reg)]= score;

for val in scores.keys():
    print ">>>>>>> ", val, scores[val];

