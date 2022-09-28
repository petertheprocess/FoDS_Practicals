# %% [markdown]
# # Practical 1 : Implementation of Linear Regression (Ridge, Lasso)
# 
# This practical has two parts. The first part is to implement and train a linear regression model using the least squares method. We will implement the model and train it on the winequality dataset using the NumPy library. Learning curves will be plotted to determine if the model is overfitting or underfitting.
# 
# In the second part, we will apply the basis expansion to the dataset and train linear regression models with regularization, i.e., Ridge and Lasso. These tasks do not need to be implemented from scratch -- We will use the functions from the scikit-learn library. During training, we use the validation data to determine the optimal hyperparameters. 
# An optional task is to implement the hyperparameter selection using the k-fold cross-validation method. Five bonus points are awarded for this task. 
# 
# 

# %% [markdown]
# We will use the following packages:

# %%
import numpy as np
import matplotlib.pyplot as plt
import _pickle as cp

# %% [markdown]
# ## Dataset
# We will use the winequality dataset for this practical. The dataset is available here:
# https://archive.ics.uci.edu/ml/datasets/Wine+Quality. 
# In order to make it easier to import the dataset, the dataset has been converted to the numpy array format and shuffled, so that we can start the practical directly. The converted dataset is available on the OLAT page.
# 
# #### Attribute Information
# Input variables (based on physicochemical tests):
# 1. fixed acidity
# 2. volatile acidity
# 3. citric acid
# 4. residual sugar
# 5. chlorides
# 6. free sulfur dioxide
# 7. total sulfur dioxide
# 8. density
# 9. pH
# 10. sulphates
# 11. alcohol
# 
# Output variable (based on sensory data):
# 
# 12. quality (score between 0 and 10)
# 
# 

# %% [markdown]
# There are two files in the dataset, one for white wine data and one for red wine data. We focus on the white wine data, which is the larger dataset. The following code loads the data:

# %%
# load the white wine dataset
# X is the feature matrix that stores the feature values of the data records
# y is the label vector that stores the labels of the data records
X, y = cp.load(open('winequality-white.pickle', 'rb'))

# check the size of the data
print("X is a matrix with the dimension {}. That is, {} data records and {} features.".format(X.shape, X.shape[0], X.shape[1]))
print("y is a vector with {} values. They are the labels of the data records in X.".format(y.shape[0]))


# %% [markdown]
# ## Understanding What We’re Predicting
# 
# First, let's look at the labels. 
# We make the bar chart below to show the distribution of labels in the dataset. 
# The y-values are integers from 3 to 9. 
# 

# %%
# the function takes the y-values in the training data as the input and makes the bar chart. 
def plot_bar_chart_score(y):
    fix, ax = plt.subplots()
    unique, counts = np.unique(y, return_counts=True)
    plt.bar(unique, counts)
    plt.xlabel('Score')
    plt.ylabel('Number of wines')
    plt.title('Distribution of scores of wines')
    plt.show()

plot_bar_chart_score(y)

# %% [markdown]
# Our goal is to train a model that can take wine records in X and predict the quality of the wines.
# We split the data into training data and test data. 
# In practice, we should sample randomly 80% of the data as training data and the rest as the test data. 
# Though, in this practical, let's all use the same split: the **first** 80% of the data for training and the remaining 20% for testing. This makes sure we can compare your results to ours.

# %% [markdown]
# ### **Task 1** 
# The first task is to write a function that can split the dataset into training and testing data according to the parameter `split_coeff`. When `split_coeff` is set to 0.8, it means the function sets the first 80% of the data as the training data and the remaining 20% as the testing data. 

# %%
# The function splits the dataset into training data and testing data.
# The parameter split_coeff is a percentage value such that
# the first split_coeff of the dataset become the training data
# and the remaining data become the test data

def split_data(X, y, split_coeff=0.8):
    # The code below is just for compilation. 
    # You need to delete it and write your own code.
    ###################################################
    ##### YOUR CODE STARTS HERE #######################
    ###################################################
    data_size = y.shape[0]

    train_index = np.random.choice(np.array(range(data_size)), 
                    size = int(split_coeff*data_size),
                    replace = False)
    test_index = np.delete(np.array(range(data_size)), train_index)

    X_train = X[train_index]
    y_train = y[train_index] 
    X_test = X[test_index]
    y_test = y[test_index]
    ###################################################
    ##### YOUR CODE ENDS HERE #########################
    ###################################################
    
    return X_train, y_train, X_test, y_test


# %% [markdown]
# Your function should return the expected output as shown in the code block below. 

# %%
X_train, y_train, X_test, y_test = split_data(X, y, 0.8) # split the data with split_coeff=0.8

# check the size of the splitted dataset
print("Shape of X_train:", X_train.shape) # expected output (3918, 11)
print("Shape of y_train:", y_train.shape) # expected output (3918,)
print("Shape of X_test:", X_test.shape) # expected output (980, 11)
print("Shape of y_test:", y_test.shape) # expected output (980,)

# %% [markdown]
# ### **Task 2** 
# 
# As a first step, we construct a trivial model that returns the mean of the y-values in the training data for any wine record in X. We use this trivial model as a baseline. The linear regression models we build later should perform better than this trivial model.

# %%
#@title
# Task 2: implement the trivial predictor
# The function computes the average value of y on the training label values
def compute_average(y_train):
    # The code below is just for compilation. 
    # You need to delete it and write your own code.
    ###################################################
    ##### YOUR CODE STARTS HERE #######################
    ###################################################
    # Hint: return the mean of y_train
    avg = np.mean(y_train)

    return avg
    ###################################################
    ##### YOUR CODE ENDS HERE #########################
    ###################################################

y_train_avg = compute_average(y_train)
print("The average of the y-values in the training data is {}".format(y_train_avg)) 

# The trivial predictor returns the average value.
def trivial_predictor(X_test, y_train_avg):
  return y_train_avg


# %% [markdown]
# ### **Task 3**
# We next evaluate the trivial predictor on the training data and test data. 
# We use mean squared error (MSE) to measure the performance of the predictor.
# The task is to implement a function that reports the mean squared error of the given predictor on the given data. 

# %%
# We next test our trivial predictor on the training data and test data. 
# Implement a function that can report the mean squared error 
# of a predictor on the given data
# Input: data and predictor
# Output: mean squared error of the predictor on the given data
def test_predictor(X, y, predictor: callable=None):
    # Apply the predictor to each row of the matrix X to get the predictions
    y_predicted = np.apply_along_axis(predictor, 1, X)

    # TODO: compute the mean squared error of y
    # The code below is just for compilation. 
    # You need to delete it and write your own code.
    ###################################################
    ##### YOUR CODE STARTS HERE #######################
    ###################################################
    mse = ((y - y_predicted)**2).mean(axis=None)
    ###################################################
    ##### YOUR CODE ENDS HERE #########################
    ###################################################
    
    return mse

# use the function test_predictor to test the trivial predictor
# we use the lambda function here to pass the function trivial predictor to the function test_predictor.
mse_trivial_predictor_train = test_predictor(X_train, y_train, lambda x: trivial_predictor(x, y_train_avg))
mse_trivial_predictor_test = test_predictor(X_test, y_test, lambda x: trivial_predictor(x, y_train_avg))

# Report the result
print('Trivial Predictor')
print('--------------------------------------------------------------------------------\n')
print('MSE (Training) = %.4f' % mse_trivial_predictor_train)
print('MSE (Testing)  = %.4f' % mse_trivial_predictor_test)

# %% [markdown]
# ## Train the Linear Model Using the Least Squares Method
# 
# Let's next train a linear regression model on the training data. 
# We use the closed form solution of the least squares estimate to compute the parameters of the linear regression model. 
# This model should perform better than the trivial predictor.

# %% [markdown]
# ### **Task 4**
# Before training the model, we need to standardize the data, i.e., transform the data so that every feature has mean 0 and variance 1. 
# 
# https://en.wikipedia.org/wiki/Standard_score
# 
# We first standardize the training data, and then apply the same transformation to the test data. That is, standardize the test data using the means and the standard deviations of the training data. 

# %%
# The task is to implement a function that can standardize the data and returns the mean and std of the data.
# Input: training data
# Output: standardize training data, standard deviations and means
def standardize_data(X):
    # TODO: compute the means and standard deviations of the data, and standardize the data
    # The code below is just for compilation. 
    # You need to replace it by your own code.
    ###################################################
    ##### YOUR CODE STARTS HERE #######################
    ###################################################
    mean = np.mean(X)
    std = np.std(X)
    X_std = (X-mean)/std
    ###################################################
    ##### YOUR CODE ENDS HERE #########################
    ###################################################
    
    return X_std, mean, std

# Standardize the training data and store the means and the stds 
X_train_std, X_train_mean, X_train_std_div = standardize_data(X_train)
print("X_train_std:", X_train_std.shape)
print("Mean:", X_train_mean)
print("Standard deviation:", X_train_std_div)

# %%
# TODO: Standardize the test data using the means and standrad deviations of the training data
###################################################
##### YOUR CODE STARTS HERE #######################
###################################################
X_test_std = standardize_data(X_test)[0]
print(X_test_std.shape)
###################################################
##### YOUR CODE ENDS HERE #########################
###################################################

# %% [markdown]
# ### **Task 5**
# Let's now train the linear model using the least-squares method. 
# We need to add the bias term to the matrix X. 

# %%
# The task is to implement the function that adds a column of ones to the first column of X
def expand_with_ones(X):
    # TODO: add a column of ones to the front of the input matrix
    # The code below is just for compilation. 
    # You need to replace it by your own code.
    ###################################################
    ##### YOUR CODE STARTS HERE #######################
    ###################################################
    dim = X.shape[0]
    X_out = np.concatenate((np.ones([dim,1]),X), axis=1)
    return X_out
    ###################################################
    ##### YOUR CODE ENDS HERE #########################
    ###################################################

# Train the linear model using the least-squares method
# The task is to implement the function that computes the parameters
def least_squares_compute_parameters(X_input, y):
    # add the bias column to the data
    X = expand_with_ones(X_input) # Q: why we need expand the X?
                    ## A: y = w0 + w1*x0 + ... + w11*x10, the expanded ones
                    ## is used to calculate the bias parameter:w0.
    # print(X.shape)
    # TODO: compute the parameters based on the expanded X and y using the least-squares method
    # The code below is just for compilation. 
    # You need to replace it by your own code.
    ###################################################
    ##### YOUR CODE STARTS HERE #######################
    ###################################################
    w = np.linalg.inv(X.T@X)@X.T@y # a@b -means- matmul(a,b)
    return w
    ##################################################
    ##### YOUR CODE ENDS HERE #########################
    ###################################################

# Apply the function to train the linear model
w = least_squares_compute_parameters(X_train_std, y_train) 
print("w:", w.shape)

# %% [markdown]
# After computing the parameters,
# we can build the linear model predictor.
# The predictor takes as input the computed parameters and the data, and predicts the labels for the input data.

# %%
# Implement the linear model predictor
# Input: test data and parameters
# Output: predicted values
def linear_model_predictor(X, w):
    # TODO: predict the labels for the input data
    # The code below is just for compilation. 
    # You need to replace it by your own code.
    ###################################################
    ##### YOUR CODE STARTS HERE #######################
    ###################################################
    # X_expand = expand_with_ones(X) 
    # in [32], the X_test_std has been expanded, so don't epand twice!
    y_predict = X @ w
    return y_predict
    ###################################################
    ##### YOUR CODE ENDS HERE #########################
    ###################################################


# %% [markdown]
# Evaluate our linear regression model predictor on the test data. 

# %%
# use the function test_predictor to evaluate the linear model predictor
mse_linear_model_predictor = test_predictor(expand_with_ones(X_test_std), y_test, lambda x: linear_model_predictor(x, w))
print("Mean squared error is {}".format(mse_linear_model_predictor))

# %% [markdown]
# ## Learning Curves
# 
# Let us check if the linear model is overfitting or underfitting. Since the dataset is somewhat large and there are only 11 features, the model shouldn't be overfitting. 
# To check it, we use the learning curves: we plot how the performance of the model changes when it is trained with increasingly more data. 

# %% [markdown]
# ### **Task 6** 
# 
# Let's first implement a function that comprises what we have implemented above. 
# The function takes as inputs the data and the split coefficient, and then
# 1. standardizes the data,
# 2. trains the linear model (compute the parameters), and
# 3. reports the MSE of the linear model predictor on both the training and the test data. 

# %%
# Input: training data and test data
# Output: mse of the linear model predictor on both the training and test data
def train_and_test(X_train, y_train, X_test, y_test):
    # TODO: implement the function 
    # The code below is just for compilation. 
    # You need to replace it by your own code.
    ###################################################
    ##### YOUR CODE STARTS HERE #######################
    ###################################################
    # Hints: use the functions you have implemented
    ## step1: prepard the data
    X_train_std = standardize_data(X_train)[0]
    X_test_std = standardize_data(X_test)[0]

    ## step2: train the model and get the parameters
    w = least_squares_compute_parameters(X_train_std, y_train)

    ## step3: evaluate the result
    mse_train = test_predictor(expand_with_ones(X_train_std), y_train, lambda x: linear_model_predictor(x, w))
    mse_test = test_predictor(expand_with_ones(X_test_std), y_test, lambda x: linear_model_predictor(x, w))
    return mse_train, mse_test
    ###################################################
    ##### YOUR CODE ENDS HERE #########################
    ###################################################

mse_train, mse_test = train_and_test(X_train, y_train, X_test, y_test)
print('Linear Model')
print('-----------------------\n')
print('MSE (Training) = %.4f' % mse_train)
print('MSE (Testing)  = %.4f' % mse_test)


# %% [markdown]
# ### **Task 7**
# 
# We are now ready to plot the learning curves. 
# 
# We train a list of models on the increasingly more training data ([20, 40, ..., 600] data records).
# For each of such model, we compute the MSEs of the model on both the training data and the test data, and store the MSEs in the lists `mse_train_v` and `mse_test`, respectively. 
# The code provided below will then plot the learning curves.
# 
# Your plot should show the two curves (roughly) meet at round 150 training data records.

# %%
mse_train_v = []
mse_test_v = []

TRAINING_SIZE_MAX = 601
TRAINING_SIZE_MIN = 20

# compute the MSE over data with sizes from TRAINING_SIZE_MIN to TRAINING_SIZE_MAX with increasing step 20
for train_size in range(TRAINING_SIZE_MIN, TRAINING_SIZE_MAX, 20):
    # TODO: 
    #   1. use the first train_size data records from the X_train and y_train as the training data
    #   2. train and compute the MSE on both training and test data using the train_and_test function
    #   3. add the computed MSE to the lists mse_train_v and mse_test_v
    # The code below is just for compilation. 
    # You need to replace it by your own code.
    ###################################################
    ##### YOUR CODE STARTS HERE #######################
    ###################################################
    mse_train, mse_test = train_and_test(X_train[:train_size],y_train[:train_size],X_test,y_test)
    
    mse_train_v.append(mse_train)
    mse_test_v.append(mse_test)
    ###################################################
    ##### YOUR CODE ENDS HERE #########################
    ###################################################
    

# The below code generates the learning curves plot
plt.plot(np.arange(TRAINING_SIZE_MIN, TRAINING_SIZE_MAX, 20), mse_train_v, 'r--', label="Training Error")
plt.plot(np.arange(TRAINING_SIZE_MIN, TRAINING_SIZE_MAX, 20), mse_test_v, 'b-', label="Test Error")
plt.legend(loc="upper right")
plt.xlabel('Dataset Size')
plt.ylabel('Mean Squared Error')
plt.show()

# %% [markdown]
# 
# 
# ```
# # 此内容为代码格式
# ```
# 
# ## Polynomial Basis Expansion and Regularisation
# 
# In this part, we will improve the linear regression model by basis expansion and regularization: 
# 1. apply the degree 2 basis expansion to the data, 
# 2. build the Ridge and Lasso models and
# 3. perform hyperparameter optimization using the validation data. 
# 
# For the hyperparameter optimization, you should set the last 20% of the training data for the purpose of validation and try lambda values in the range [10^-4, 10^-3, 10^-2, 10^-1, 1, 10, 100]. 
# 
# We will use the scikit-learn library. We have imported the necessary functions for you. You can import other scikit-learn functions if you think they are useful. The documentation is available here: http://scikit-learn.org/stable/modules/classes.html

# %%
# import the preprocessing libs for standarization and basis expansion
from sklearn.preprocessing import StandardScaler, PolynomialFeatures 

# Ridge and Lasso linear model
from sklearn.linear_model import Ridge, Lasso 

# %% [markdown]
# ### **Task 8**
# Let's implement the function for expanding the basis of the dataset. 
# 
# Hints: use `PolynomialFeatures`

# %%
def expand_basis(X, degree):
    # TODO: expand the basis of X for the input degree
    # The code below is just for compilation. 
    # You need to replace it by your own code.
    ###################################################
    ##### YOUR CODE STARTS HERE #######################
    ###################################################
    # Hints: use the function PolynomialFeatures
    poly = PolynomialFeatures(degree)
    X_expand = poly.fit_transform(X)
    return X_expand
    ###################################################
    ##### YOUR CODE ENDS HERE #########################
    ###################################################

# %% [markdown]
# ### **Task 9**
# We need to prepare the data.
# We first expand and standardize the data,
# and then split the training data to training data and validation data.
# We use the last 20% of the training data as the validation data.
# 
# Hints: use `StandardScaler` and `std_scaler` to standardize the data

# %%
def prepare_data(X, y, degree):
    # TODO: the training, test and validation data using the expanded dataset.
    # The code below is just for compilation. 
    # You need to replace it by your own code.
    ###################################################
    ##### YOUR CODE STARTS HERE #######################
    ###################################################
    # Hints: follow the steps     
    # 1. split the data (X, y) into training data (X_train, y_train) and test data (X_test, y_test)
    X_train,y_train,X_test,y_test = split_data(X,y)    
    # 2. standardize the training data and do the same transformation to the test data
    X_train = standardize_data(X_train)[0]
    X_test = standardize_data(X_test)[0]
    # 3. expand the basis of the training data and test data
    X_train = expand_basis(X_train,2)
    X_test = expand_basis(X_test,2)
    # 4. split the expanded training data into training data (X_train_n, y_train_n) and validation data (X_train_v, y_train_v)
    X_train_n,y_train_n,X_train_v,y_train_v = split_data(X_train,y_train) 

    return X_train, y_train, X_train_n, y_train_n, X_train_v, y_train_v, X_test, y_test
    ###################################################
    ##### YOUR CODE ENDS HERE #########################
    ###################################################

X_train, y_train, X_train_n, y_train_n, X_train_v, y_train_v, X_test, y_test = prepare_data(X, y, 2) # here we expand the dataset with degree 2

# check the size of the splitted dataset
print("Shape of X_train_n:", X_train_n.shape) # expected output (3134, 78)
print("Shape of y_train_n:", y_train_n.shape) # expected output (3134,)
print("Shape of X_train_v:", X_train_v.shape) # expected output (784, 78)
print("Shape of y_train_v:", y_train_v.shape) # expected output (784,)
print("Shape of X_test:", X_test.shape) # expected output (980, 78)
print("Shape of y_test:", y_test.shape) # expected output (980,)

# %% [markdown]
# ### **Task 10**
# We have prepared the training data and the validation data. 
# We can now use the validation data to select the optimal hyperparameters for the Ridge and Lasso models. 
# We use the Ridge and Lasso models from scikit-learn: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
# 
# We train Ridge or Lasso models with different lambda values and check their performance on the validation data, and select the lambda values that yield the best performance. 

# %%
from sklearn import linear_model
# The function takes the training and validation data as inputs, and 
# returns the lambda value that results the minimal mse
# We use is_ridge to indicate which the model is considered.
# is_ridge = True indicates Ridge while is_ridge = False indicates Lasso
def choose_hyper_param(X_train_n, y_train_n, X_train_v, y_train_v, is_ridge: bool):
    mse_arr = []
    lam_arr = []

    # Try lambda values from 10^-4 to 10^2. 
    # Record the mse and the lambda values in mse_arr and lam_arr
    # The code below is just for compilation. 
    # You need to replace it by your own code.
    ###################################################
    ##### YOUR CODE STARTS HERE #######################
    ###################################################
    for pow_lam in range(-4, 3):
        lam = 10 ** pow_lam
        if(is_ridge):
            reg = linear_model.Ridge(alpha=lam)
        else:
            reg = linear_model.Lasso(alpha=lam)
        reg.fit(X_train_n,y_train_n)
        y_predicted_v = reg.predict(X_train_v) 
        mse = ((y_train_v - y_predicted_v)**2).mean(axis=None)
        # compute the mse for this lam
        mse_arr.append(mse) 
        lam_arr.append(lam) 
    ###################################################
    ##### YOUR CODE ENDS HERE #########################
    ###################################################


    # get the index of the lambda value that has the minimal use
    lambda_idx_min = np.argmin(np.array(mse_arr))

    # plot of the lambda values and their mse
    plt.figure()
    plt.semilogx(lam_arr, mse_arr)

    # return the optimal lambda value
    return lam_arr[lambda_idx_min]

# call the function to choose the lambda for Ridge and Lasso
lam_ridge = choose_hyper_param(X_train_n, y_train_n, X_train_v, y_train_v, True)
lam_lasso = choose_hyper_param(X_train_n, y_train_n, X_train_v, y_train_v, False)

print("Ridge lambda:", lam_ridge)
print("Lasso lambda:", lam_lasso)

# %% [markdown]
# ### **Task 11**:
# Once we get the optimal lambdas for Ridge and Lasso, we train these models using these lambdas on the full training data, and then report their performance.

# %%
# TODO: train the Ridge and Lasso models using the optimal parameters, and
#       report their MSE
###################################################
##### YOUR CODE STARTS HERE #######################
###################################################
# Hints: train these models on the full training data
reg_ridge = linear_model.Ridge(lam_ridge)
reg_ridge.fit(X_train,y_train)
mse_ridge_train = ((y_train - reg_ridge.predict(X_train))**2).mean(axis=None)
mse_ridge_test = ((y_test - reg_ridge.predict(X_test))**2).mean(axis=None)

reg_lasso = linear_model.Lasso(lam_lasso)
reg_lasso.fit(X_train,y_train)
mse_lasso_train = ((y_train - reg_lasso.predict(X_train))**2).mean(axis=None)
mse_lasso_test = ((y_test - reg_lasso.predict(X_test))**2).mean(axis=None)
###################################################
##### YOUR CODE ENDS HERE #########################
###################################################

# Report the result
print('For Ridge Regression with using degree %d polynomial expansion and lambda = %.4f' % (2, lam_ridge))
print('--------------------------------------------------------------------------------\n')
print('MSE (Training) = %.4f' % mse_ridge_train)
print('MSE (Testing)  = %.4f' % mse_ridge_test)

print('\n\nFor Lasso with using degree %d polynomial expansion and lambda = %.4f' % (2, lam_lasso))
print('---------------------------------------------------------------------\n')
print('MSE (Training) = %.4f' % mse_lasso_train)
print('MSE (Testing)  = %.4f' % mse_lasso_test)

# %% [markdown]
# ## Optional: Try Larger Degrees using K-fold Cross Validation
# 
# 

# %% [markdown]
# ### **Task 12**
# This is an optional task, which worths 5 bonus points.
# 
# The task is to try basis expansions with higher degrees (up to degree 4) and find the degree that results the best performance. 
# Instead of always using the same validation data, we use k-fold cross-validation to find the optimal hyperparameters. 
# 
# Your task is to report the optimal hyperparameters (the basis expansion degree and the lambdas) and the MSEs of the Ridge and Lasso when using these optimal hyperparameters. 
# 
# Hints: Use `KFold` to do this automatically. 

# %%
# KFold
from cgi import test
from sklearn.model_selection import KFold

###################################################
##### YOUR CODE STARTS HERE #######################
###################################################
# step1 
#   choose hyperparameter by kFold


def choose_hyper_param_KFlod(X_train, y_train, is_ridge: bool, n_splits):
    kf = KFold(n_splits)
    mse_mean_arr = []
    lam_arr = []
    for pow_lam in range(-4, 3):
        mse_arr = []
        lam = 10 ** pow_lam

        for train_index, validate_index in kf.split(X_train):
            X_train_n, X_train_v = X_train[train_index], X_train[validate_index]
            y_train_n, y_train_v = y_train[train_index], y_train[validate_index]
            
            if(is_ridge):
                reg = linear_model.Ridge(alpha=lam)
            else:
                reg = linear_model.Lasso(alpha=lam)
            reg.fit(X_train_n,y_train_n)
            y_predicted_v = reg.predict(X_train_v) 
            mse = ((y_train_v - y_predicted_v)**2).mean(axis=None)
            # compute the mse for this lam
            mse_arr.append(mse)

        mse_mean = np.mean(mse_arr)
        mse_mean_arr.append(mse_mean)
        lam_arr.append(lam)

    # get the index of the lambda value that has the minimal use
    lambda_idx_min = np.argmin(np.array(mse_mean_arr))
    # return the optimal lambda value
    return lam_arr[lambda_idx_min]

mse_ridge_train_arr = []
mse_ridge_test_arr = []
mse_lasso_train_arr = []
mse_lasso_test_arr = []

for degree in range(2,5): # degree 2, 3 ,4
    print("---------------degree %d polynomial expansion---------------" % (degree))
    X_train,y_train,_,_,_,_,X_test,y_test = prepare_data(X, y, degree)
    # apply 10-fold algorithm to choose hyper param for Ridge and Lasso
    lam_ridge = choose_hyper_param_KFlod(X_train, y_train, True, 10)
    lam_lasso = choose_hyper_param_KFlod(X_train, y_train, False, 10)

    print("Ridge lambda:", lam_ridge)
    print("Lasso lambda:", lam_lasso)    
    # step2
    #  Model training and the MSEs results
    reg_ridge = linear_model.Ridge(lam_ridge)
    reg_ridge.fit(X_train,y_train)
    mse_ridge_train = ((y_train - reg_ridge.predict(X_train))**2).mean(axis=None)
    mse_ridge_test = ((y_test - reg_ridge.predict(X_test))**2).mean(axis=None)
    mse_ridge_train_arr.append(mse_ridge_train)
    mse_ridge_test_arr.append(mse_ridge_test)

    reg_lasso = linear_model.Lasso(lam_lasso)
    reg_lasso.fit(X_train,y_train)
    mse_lasso_train = ((y_train - reg_lasso.predict(X_train))**2).mean(axis=None)
    mse_lasso_test = ((y_test - reg_lasso.predict(X_test))**2).mean(axis=None)
    mse_lasso_train_arr.append(mse_lasso_train)
    mse_lasso_test_arr.append(mse_lasso_test)

    # Report the result
    print('For Ridge Regression with using degree %d polynomial expansion and lambda = %.4f' % (degree, lam_ridge))
    print('--------------------------------------------------------------------------------\n')
    print('MSE (Training) = %.4f' % mse_ridge_train)
    print('MSE (Testing)  = %.4f' % mse_ridge_test)

    print('\n\nFor Lasso with using degree %d polynomial expansion and lambda = %.4f' % (degree, lam_lasso))
    print('---------------------------------------------------------------------\n')
    print('MSE (Training) = %.4f' % mse_lasso_train)
    print('MSE (Testing)  = %.4f' % mse_lasso_test)


###################################################
##### YOUR CODE ENDS HERE #########################
###################################################

# %%
# The below code generates the learning curves plot
plt.figure()
plt.plot(np.arange(2,5), mse_ridge_train_arr, 'ro--', label="Ridge Training Error",markersize=4)
plt.plot(np.arange(2,5), mse_ridge_test_arr, 'ro-', label="Ridge Test Error",markersize=4)
plt.plot(np.arange(2,5), mse_lasso_train_arr, 'bo--', label="Lasso Test Error",markersize=4)
plt.plot(np.arange(2,5), mse_lasso_test_arr, 'bo-', label="Lasso Test Error",markersize=4)

plt.legend(loc="upper right")
plt.xlabel('Polynomial Expansion Degree')
plt.ylabel('Mean Squared Error')
plt.show()


