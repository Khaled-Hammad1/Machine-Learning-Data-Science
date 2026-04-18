import pandas
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import title
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression , Ridge , LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.metrics import mean_squared_error


'''
Prepared by:  
Khaled Hammad       ID: 1220857    Sec: 1 
Mohammad Shamasneh  ID: 1220092    Sec: 1 
'''

#==============================================================| task 1 |===================================================



np.random.seed(7) #I put it to make the random is static in each compile
n_samples = 25
x_train = np.random.rand(n_samples)# x in [0,1]
noise = np.random.uniform(-0.3, 0.3, n_samples)
y_train = np.sin(5 * np.pi * x_train) + noise    # yt = sin(5πx) + noise
#true function
x_plot = np.linspace(0, 1, 500)
y_true = np.sin(5 * np.pi * x_plot) #The original function(Y target)

##Part 1 - A

poly = PolynomialFeatures(degree=9, include_bias=True)
X_trainP = poly.fit_transform(x_train.reshape(-1, 1)) #reshape is to make it as matrix
X_plotP = poly.transform(x_plot.reshape(-1, 1))
lam = [0.0,0.000000000001,0.000001, 0.01, 5]
# MSE_lam = []
plt.figure(figsize=(10, 6))

for l in lam:
    ridge = Ridge(alpha=l, fit_intercept=False)# fit_intercept=False because PolynomialFeatures contain bias
    ridge.fit(X_trainP, y_train)
    y_pred = ridge.predict(X_plotP)
    # mse = mean_squared_error(y_true, y_pred)
    # MSE_lam.append(mse)

    plt.plot(x_plot, y_pred, label=f'λ={l}')

plt.scatter(x_train, y_train, color='black', label='Training data')
plt.plot(x_plot, y_true, linestyle='--', label='True sin(5πx)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Part A: Ridge Regression (degree=9)')
plt.legend()
plt.tight_layout()
plt.show()

# best_idx = int(np.argmin(MSE_lam))
# print("Part A - Best λ:", lam[best_idx], "with MSE:", MSE_lam[best_idx])

################################################################################################################33
# Part 1 - B

def make_rbf_features(x, centers, lam, add_bias=True):
    x = x.reshape(-1, 1)            # (n, 1)
    centers = centers.reshape(1, -1)  # (1, m)

    rbf = np.exp(- (x - centers) ** 2 / lam)   # (n, m)

    if add_bias:
        rbf = np.concatenate([np.ones((x.shape[0], 1)), rbf], axis=1)
    return rbf

rbf_counts = [1, 5, 10, 50]

plt.figure(figsize=(10, 8))
# plt.scatter(x_train, y_train, color='black', label='Training data')

for m in rbf_counts:
    centers = np.linspace(0, 1, m)

    if m==1:
        centers = np.array([0.5])
        lam = 0.5
    if m > 1:
        spacing = centers[1] - centers[0]
        lam = spacing**2


    Z_train = make_rbf_features(x_train, centers, lam, add_bias=True)

    linreg = LinearRegression(fit_intercept=False)
    linreg.fit(Z_train, y_train)

    Z_plot = make_rbf_features(x_plot, centers, lam, add_bias=True)
    y_rbf_plot = linreg.predict(Z_plot)

    # mse_rbf = mean_squared_error(y_true, y_rbf_plot)

    plt.plot(x_plot, y_rbf_plot, label=f'{m} RBFs')
plt.scatter(x_train, y_train, color='black', label='Training data')
plt.plot(x_plot, y_true, linestyle='--', label='True sin(5πx)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Part B: Non-linear Regression with RBF basis functions')
plt.legend()
plt.tight_layout()
plt.show()













#==============================================================| task 2 |===================================================




######################################| read the data |############################################################

data = pandas.read_csv("customer_data.csv")

######################################| Handling Missing Data |############################################################

data['Age'] = data['Age'].fillna(data['Age'].median())
data['Tenure'] = data['Tenure'].fillna(data['Tenure'].median())
data['SupportCalls'] = data['SupportCalls'].fillna(data['SupportCalls'].median())
data['Income'] = data.groupby('ProductType')['Income'].transform(lambda x: x.fillna(x.median()))

######################################| Clean Data from outlier |############################################################

for col in ['Age', 'Tenure', 'SupportCalls','Income']:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3- Q1
    lowerBound = Q1 - 1.5 * IQR
    upperBound = Q3 + 1.5 * IQR
    medianValue = data[col].median()
    data[col] =data[col].apply(lambda x: medianValue if x < lowerBound or x > upperBound else x) # we change the outlier by median of column instead of remove all row

######################################| Scaling (Min-Max Normalization) |############################################################

for col in ['Age', 'Tenure', 'SupportCalls','Income']:
    minv = data[col].min()
    maxv = data[col].max()
    data[col] = (data[col]-minv)/(maxv-minv)

'''
------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------
'''


X = data.drop(["ChurnStatus", "CustomerID" ,], axis=1) # this mean thee data without ChurnStatus is input , axis =1 mean we remove colums / axis =0 mean rows
y = data["ChurnStatus"] # ChurnStatus is the target value 

X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=2500, random_state=42, shuffle=True)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, train_size=500, random_state=42, shuffle=True)
'''
above we split the data to 2500 training samples, 500 validation samples, and 500 test samples. 
'''


#-------



def evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test):
    results = {}
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)
    
    results["Accuracy_Train"] = accuracy_score(y_train, y_pred_train)
    results["Accuracy_Val"] = accuracy_score(y_val, y_pred_val)
    results["Accuracy_Test"] = accuracy_score(y_test, y_pred_test)
    
    results["Precision_Train"] = precision_score(y_train, y_pred_train)
    results["Precision_Val"] = precision_score(y_val, y_pred_val)
    results["Precision_Test"] = precision_score(y_test, y_pred_test)
    
    results["Recall_Train"] = recall_score(y_train, y_pred_train)
    results["Recall_Val"] = recall_score(y_val, y_pred_val)
    results["Recall_Test"] = recall_score(y_test, y_pred_test)
    
    return results


def print_metrics(results , degree ):
    if (degree == 1) :
        print(f"\n===== Linear Model =====")
    else :
        print(f"\n===== Model (Degree = {degree}) =====")
    metrics_order = ["Accuracy", "Precision", "Recall"]
    sets = ["Train", "Val", "Test"]
    for metric in metrics_order:
        print(f"\n--- {metric} ---")
        for s in sets:
            key = f"{metric}_{s}"
            if key in results:
                print(f"{s}: {results[key]:.4f}") 
        
    print("============================================\n")







model_linear = LogisticRegression(max_iter=500) #create linear LogisticRegression 
model_linear.fit(X_train, y_train) # train
metrics_linear = evaluate(model_linear, X_train, y_train, X_val, y_val, X_test, y_test) # evaluate Accuracy (Train / Val / Test) Precision Recall
print_metrics(metrics_linear,1)
'''
coef = model_linear.coef_
intercept = model_linear.intercept_
print("Coefficients:", coef)
print("Intercept:", intercept)
'''



degrees = [2, 5, 9]
metrics_poly = {}
models_poly  = {}

for d in degrees:
    poly = PolynomialFeatures(d)
    X_train_p = poly.fit_transform(X_train)
    X_val_p = poly.transform(X_val)
    X_test_p = poly.transform(X_test)
    
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train_p, y_train)
    models_poly[d] = (model, poly)
    metrics_poly[d] = evaluate(model, X_train_p, y_train, X_val_p, y_val, X_test_p, y_test)
    ("-------------------------------------------------------------------------------------------------------")
    print_metrics(metrics_poly[d],d)
    
    
""" print(f"Degree {d} : \n {metrics_poly[d]} ")
    print(poly.get_feature_names_out(X_train.columns))
    print(model.coef_)
    print("-------------------------------------------------------------------------------------------------------")
    print(model.intercept_)
    """


metrics_poly[1] = metrics_linear

best_degree =  max(metrics_poly,key=lambda d: metrics_poly[d]["Recall_Val"])

best_model, best_poly = models_poly[best_degree]

print("\nBest Model Degree =", best_degree)
print(print_metrics(metrics_poly[best_degree],best_degree))




X_test_best = best_poly.transform(X_test)

y_scores = best_model.predict_proba(X_test_best)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", linewidth=2)
plt.plot([0,1], [0,1], "--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve - Logistic Regression (Degree {best_degree})")
plt.legend()
plt.grid()
plt.show()



