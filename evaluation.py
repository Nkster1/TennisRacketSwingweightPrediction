
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


def calculate_metrics(y_true, y_pred):
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    med_ae = metrics.median_absolute_error(y_true, y_pred)
#    mape = metrics.mean_absolute_percentage_error(y_true, y_pred)

    print("mean_absolute_error", mae)
    #print("mean_squared_error", mse)
    print("RMSE", np.sqrt(mse))
    print("median_absolute_error", med_ae)
    #print("mean_absolute_percentage_error", mape)

def calculate_mad(x):
    return np.median(np.absolute(x - np.median(x)))

def evaluate_model(model, X_train, y_train,X_test, y_test):
    model = model.fit(X_train, y_train)
    print("train_score", model.score(X_train, y_train))
    print("test_score", model.score(X_test, y_test))
    y_pred = model.predict(X_test)
    calculate_metrics(y_test, y_pred)

    difference = y_test - y_pred
    plt.hist(difference, bins=70)
#     print("mean of difference: ", np.mean(difference))
#     print("sd of difference: ", np.std(difference))

    med = np.median(difference)
    mad = calculate_mad(difference)
#     print("median of difference: ", med)
#     print("mean absolute difference: ", mad)

    lower = difference < med - 3 * mad
    upper = difference > med + 3 * mad



    return model


def predict(model, flex, weight, balance):
    model = model.fit(X_train,y_train)
    sample = np.array([[ flex, weight, balance]])
    try:
        prediction = model.predict( np.array([[weight, balance,  flex]]))
    except:
        prediction = model.predict( np.array([[ weight, balance]]))


    return prediction
