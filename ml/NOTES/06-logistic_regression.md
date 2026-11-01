# Logistic Regression

## 1. What is Logistic Regression?

Logistic Regression is a **supervised machine learning algorithm** mainly used for classification problems.

It predicts the probability that an input belongs to a particular class.

**Example:**
- Input: Number of hours studied.
- Output: Probability that a student passes an exam.
- If probability ≥ 0.5 → Pass (1).
- If probability < 0.5 → Fail (0).

Despite its name, Logistic Regression is a classification algorithm, not a regression algorithm.

## 2. How Does It Work?

Logistic Regression works in two major steps:

1. Calculate a weighted sum of input features using a linear equation.
2. Pass that result through the sigmoid function to obtain a probability between 0 and 1.

### Step 1: Linear Equation

For one feature:

\[
z = \beta_0 + \beta_1x
\]

For multiple features:

\[
z = \beta_0 + \beta_1x_1 + \cdots + \beta_nx_n
\]

Where:
- \(x_i\) = input features
- \(\beta_i\) = model coefficients (learned parameters)
- \(\beta_0\) = intercept
- \(z\) = linear output

### Step 2: Sigmoid Function

The sigmoid function converts the linear output into a probability.

\[
\sigma(z) = \frac{1}{1+e^{-z}}
\]

Its output always lies between 0 and 1.

### Step 3: Classification

Using a default threshold of 0.5:

- Probability ≥ 0.5 → Class 1
- Probability < 0.5 → Class 0

The threshold can be adjusted depending on the application.

## 3. Understanding the Sigmoid Function

The sigmoid function produces an S-shaped curve.

Important properties:
- Output range: (0, 1).
- At z = 0, output = 0.5.
- For large positive z, output approaches 1.
- For large negative z, output approaches 0.

## 4. Probability and Odds

Logistic Regression models the log-odds of the positive class.

**Odds:**

\[
Odds = \frac{P(Y=1)}{1-P(Y=1)}
\]

**Log-odds (Logit):**

\[
\log\left(\frac{p}{1-p}\right)
= \beta_0+\beta_1x_1+\cdots+\beta_nx_n
\]

A coefficient represents the change in log-odds for a one-unit increase in its feature, keeping other features constant.

The odds are multiplied by \(e^{\beta_i}\) for a one-unit increase in feature \(x_i\).

## 5. Cost Function (Important)

Logistic Regression generally uses **Log Loss (Binary Cross-Entropy)** instead of Mean Squared Error.

\[
J = -\frac{1}{m}\sum_{i=1}^{m}
[y_i\log(p_i)+(1-y_i)\log(1-p_i)]
\]

Where:
- \(m\) = number of training samples
- \(y_i\) = actual class (0 or 1)
- \(p_i\) = predicted probability

The model learns its coefficients by minimizing this loss, commonly using gradient-based optimization.

**Why not ordinary MSE?**

MSE is not the usual choice for logistic regression because it does not give the same convenient convex optimization objective for this model. Binary cross-entropy is the standard choice.

## 6. Types of Logistic Regression

| Type | Purpose |
|---|---|
| Binary Logistic Regression | Two classes (0 or 1) |
| Multinomial Logistic Regression | Three or more unordered classes |
| Ordinal Logistic Regression | Ordered categories |

Example:
- Binary: Spam / Not Spam
- Multinomial: Cat / Dog / Horse
- Ordinal: Low / Medium / High

## 7. Regularization

Regularization helps prevent overfitting by penalizing large coefficients.

- **L1 (Lasso):** Can make some coefficients exactly zero, helping with feature selection.
- **L2 (Ridge):** Shrinks coefficients towards zero but generally does not make them exactly zero.

Scikit-learn's LogisticRegression uses L2 regularization by default.

## 8. Python Implementation Using Scikit-learn

### Step 1: Import Libraries

```python
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
```

### Step 2: Create Dataset

Example: Predict whether a student passes based on study hours and attendance.

```python
X = np.array([
    [1, 40],
    [2, 45],
    [3, 50],
    [4, 55],
    [5, 60],
    [6, 65],
    [7, 75],
    [8, 80],
    [9, 85],
    [10, 90]
])

y = np.array([
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1
])
```

Here:
- X contains input features.
- y contains target labels.
- 0 = Fail
- 1 = Pass

### Step 3: Split Dataset

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

- 80% training data.
- 20% testing data.
- `random_state` makes the split reproducible.
- `stratify=y` preserves class proportions.

### Step 4: Create and Train Model

```python
model = make_pipeline(
    StandardScaler(),
    LogisticRegression()
)

model.fit(X_train, y_train)
```

`make_pipeline` ensures the scaler is fitted only on training data and applies the same transformation to test and future data.

### Step 5: Make Predictions

```python
y_pred = model.predict(X_test)

print("Predictions:", y_pred)
```

### Step 6: Predict Probabilities

```python
probabilities = model.predict_proba(X_test)

print(probabilities)
```

`predict_proba()` returns probabilities for each class.

For binary classification:
- Column 0 = Probability of class 0.
- Column 1 = Probability of class 1.

### Step 7: Evaluate Model

```python
print("Accuracy:", accuracy_score(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
```

**Important:** This is a tiny illustrative dataset, not a reliable real-world model. Use a sufficiently large representative dataset for actual evaluation.

## 9. Important Hyperparameters

| Hyperparameter | Meaning |
|---|---|
| C | Inverse of regularization strength |
| penalty | Regularization type (L1, L2, etc.) |
| solver | Optimization algorithm |
| max_iter | Maximum optimization iterations |
| class_weight | Adjusts weights for different classes |

**Remember:** Smaller C means stronger regularization; larger C means weaker regularization.

## 10. Advantages

- Simple and easy to implement.
- Computationally efficient.
- Provides class probabilities.
- Works well when the relationship between features and log-odds is approximately linear.
- Supports regularization.

## 11. Disadvantages

- Assumes a linear relationship between features and log-odds.
- May perform poorly on complex nonlinear patterns without feature engineering.
- Sensitive to multicollinearity.
- Can be affected by outliers and irrelevant features.
- May struggle with highly imbalanced datasets without suitable handling.

## 12. Logistic Regression vs Linear Regression

| Linear Regression | Logistic Regression |
|---|---|
| Used for regression | Used for classification |
| Predicts continuous values | Predicts class probabilities |
| Uses linear output | Uses sigmoid (binary case) |
| Common loss: MSE | Common loss: Log Loss |
| Output is unbounded | Probability is between 0 and 1 |

## 13. Common Interview Questions

**Q1. Why is Logistic Regression called regression if it is used for classification?**

Because it models the log-odds as a linear combination of input features. It then converts that result into a probability for classification.

**Q2. Why is sigmoid used?**

It converts any real-valued input into a probability between 0 and 1.

**Q3. What is the default classification threshold?**

0.5 in the usual binary classification setting.

**Q4. Can Logistic Regression handle multiple classes?**

Yes. Multinomial Logistic Regression supports multiple classes. One-vs-Rest is another strategy.

**Q5. What is the difference between predict() and predict_proba()?**

- `predict()` returns predicted class labels.
- `predict_proba()` returns predicted class probabilities.

**Q6. Why is feature scaling useful?**

It helps optimization and makes regularization behave more consistently across features with different scales.

**Q7. Is Logistic Regression a linear model?**

Yes. It is linear in its features with respect to log-odds. Its probability output follows a nonlinear sigmoid curve.

**Q8. What happens if the threshold is increased?**

The model generally predicts fewer positive cases, increasing precision and potentially reducing recall.

## 14. Quick Revision

- Logistic Regression = Supervised Classification.
- Sigmoid function converts linear output into probability.
- Probability range = 0 to 1.
- Default threshold = 0.5.
- Common loss = Binary Cross-Entropy / Log Loss.
- Optimization = Gradient-based methods.
- Regularization = L1 and L2.
- `fit()` = Train model.
- `predict()` = Predict class.
- `predict_proba()` = Predict probability.

## 15. Preparation Priority

| Priority | Topics |
|---|---|
| Must Know | Sigmoid, probability, classification threshold, implementation |
| Must Know | Log Loss, coefficients, predict vs predict_proba |
| Important | Regularization, odds, log-odds |
| Important | Multiclass classification, evaluation metrics |
| Advanced | Gradient descent derivation, solver differences, class imbalance |