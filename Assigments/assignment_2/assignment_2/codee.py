import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error



np.random.seed(7) #I put it to make the random is static in each compile
n_samples = 25
x_train = np.random.rand(n_samples)# x_i in [0,1]
noise = np.random.uniform(-0.3, 0.3, n_samples)
y_train = np.sin(5 * np.pi * x_train) + noise    # y_i = sin(5πx_i) + noise
#true function
x_plot = np.linspace(0, 1, 500)
y_true = np.sin(5 * np.pi * x_plot) #The original function(Y target)
#plt.plot(x_plot, y_true, label=f'λ={lam}, MSE={mse:.3f}')
#plt.show()



##Part 1 - A

degree_p = 9

poly = PolynomialFeatures(degree=degree_p, include_bias=True)
X_train_poly = poly.fit_transform(x_train.reshape(-1, 1))
X_plot_poly = poly.transform(x_plot.reshape(-1, 1))
lambdas = [0.0, 0.01, 0.1, 1.0, 10.0]
MSE_per_lambda = []

plt.figure(figsize=(10, 6))
plt.scatter(x_train, y_train, color='black', label='Training data')
# plt.show()

for lam in lambdas:
    ridge = Ridge(alpha=lam, fit_intercept=False)# fit_intercept=False because PolynomialFeatures contain bias
    ridge.fit(X_train_poly, y_train)

    y_pred_plot = ridge.predict(X_plot_poly)
    mse = mean_squared_error(y_true, y_pred_plot)
    MSE_per_lambda.append(mse)

    plt.plot(x_plot, y_pred_plot, label=f'λ={lam}, MSE={mse:.3f}')

plt.plot(x_plot, y_true, linestyle='--', label='True sin(5πx)')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Part A: Ridge Regression (degree={degree_p})')
plt.legend()
plt.tight_layout()
plt.show()

best_idx = int(np.argmin(MSE_per_lambda))
print("Part A - Best λ:", lambdas[best_idx], "with MSE:", MSE_per_lambda[best_idx])





## Part 1 - B

def make_rbf_features(x, centers, lam, add_bias=True):
    """
    x: shape (n,)
    centers: shape (m,)
    returns Z of shape (n, m [+1 if bias])

    RBF as in the slide: exp( - (x - α)^2 / λ )
    """
    x = x.reshape(-1, 1)            # (n, 1)
    centers = centers.reshape(1, -1)  # (1, m)

    rbf = np.exp(- (x - centers) ** 2 / lam)   # (n, m)

    if add_bias:
        rbf = np.concatenate([np.ones((x.shape[0], 1)), rbf], axis=1)
    return rbf


rbf_counts = [1, 5, 10, 50]

plt.figure(figsize=(10, 8))
plt.scatter(x_train, y_train, color='black', label='Training data')

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

    mse_rbf = mean_squared_error(y_true, y_rbf_plot)

    plt.plot(x_plot, y_rbf_plot, label=f'{m} RBFs, MSE={mse_rbf:.3f}')

plt.plot(x_plot, y_true, linestyle='--', label='True sin(5πx)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Part B: Non-linear Regression with RBF basis functions')
plt.legend()
plt.tight_layout()
plt.show()






