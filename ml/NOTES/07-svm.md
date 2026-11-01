# Support Vector Machine (SVM)

## 1. What is SVM?

Support Vector Machine (SVM) is a **supervised machine learning algorithm** mainly used for classification, but it can also perform regression.

Its main objective is to find the best decision boundary (hyperplane) that separates different classes.

**Example:**
Suppose we have two classes:
- Class 0: Not Spam
- Class 1: Spam

SVM finds a decision boundary that separates these classes while maximizing the margin between them.

## 2. Important Terminology

### Hyperplane
A decision boundary that separates different classes.

For a two-dimensional dataset, it is a line.

For a three-dimensional dataset, it is a plane.

For higher dimensions, it is called a hyperplane.

Mathematically:

\[
w^Tx+b=0
\]

Where:
- \(w\) = weight vector
- \(x\) = input feature vector
- \(b\) = bias

### Support Vectors
Support vectors are the data points closest to the decision boundary.

They are important because they determine the position and orientation of the optimal hyperplane.

### Margin
Margin is the distance between the decision boundary and the nearest data points from both classes.

SVM tries to maximize this margin.

For a linear SVM, the margin width is:

\[
\text{Margin}=\frac{2}{||w||}
\]

Therefore, maximizing the margin is equivalent to minimizing:

\[
\frac{1}{2}||w||^2
\]

subject to correct classification constraints in the hard-margin case.

## 3. How Does SVM Work?

1. Take the training dataset.
2. Identify possible decision boundaries.
3. Find the nearest data points from each class.
4. Select the boundary with the maximum margin.
5. Use that boundary to classify new data points.

### Hard Margin SVM

Used when the data is perfectly linearly separable.

It does not allow classification errors or margin violations.

Optimization objective:

\[
\min \frac{1}{2}||w||^2
\]

Subject to:

\[
y_i(w^Tx_i+b)\geq 1
\]

### Soft Margin SVM

Used when data is noisy or not perfectly linearly separable.

It allows some margin violations using slack variables \(\xi_i\).

Optimization objective:

\[
\min \frac{1}{2}||w||^2+C\sum_{i=1}^{m}\xi_i
\]

Subject to:

\[
y_i(w^Tx_i+b)\geq 1-\xi_i
\]

\[
\xi_i\geq 0
\]

Here, **C** controls the trade-off between maximizing the margin and penalizing classification errors.

- High C: Penalizes errors more heavily; may create a narrower margin and overfit.
- Low C: Allows more violations; generally creates a wider margin and stronger regularization.

## 4. Types of SVM

### 1. Linear SVM

Used when classes can be separated approximately using a straight line or hyperplane.

### 2. Non-Linear SVM

Used when classes cannot be separated effectively using a straight line.

It uses the **Kernel Trick** to handle nonlinear decision boundaries.

## 5. Kernel Trick (Very Important)

A kernel function allows SVM to learn nonlinear decision boundaries by implicitly computing similarities in a higher-dimensional feature space.

Instead of explicitly transforming the data into higher dimensions, it computes inner products in that space.

### Common Kernel Functions

| Kernel | Use |
|---|---|
| Linear | Linearly separable or approximately linear data |
| Polynomial | Nonlinear relationships with polynomial patterns |
| RBF (Gaussian) | Flexible nonlinear boundaries |
| Sigmoid | Similarity based on a sigmoid function |

### RBF Kernel

A commonly used kernel:

\[
K(x_i,x_j)=e^{-\gamma||x_i-x_j||^2}
\]

Where:
- \(x_i,x_j\) = data points
- \(\gamma\) = controls how far the influence of a training point reaches

**Gamma:**
- High gamma: Each point has a smaller influence area; boundary can become complex and overfit.
- Low gamma: Each point has a larger influence area; boundary becomes smoother and may underfit.

## 6. Important Hyperparameters

| Hyperparameter | Meaning |
|---|---|
| C | Controls regularization and penalty for violations |
| kernel | Specifies the kernel function |
| gamma | Controls influence of points for RBF and other applicable kernels |
| degree | Degree of polynomial kernel |
| probability | Enables probability estimates in SVC |

**Remember:**
- High C → less tolerance for errors.
- Low C → more tolerance for errors.
- High gamma → more complex boundary.
- Low gamma → smoother boundary.

## 7. Python Implementation Using Scikit-learn

We will use the Iris dataset to classify flowers into different species.

### Step 1: Import Libraries

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
```

### Step 2: Load Dataset

```python
data = load_iris()

X = data.data
y = data.target

print("Features:", X.shape)
print("Labels:", y.shape)
```

- X contains flower measurements.
- y contains the flower species labels.
- There are three classes.

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

### Step 4: Create SVM Model

```python
model = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", C=1.0, gamma="scale")
)
```

Explanation:
- `StandardScaler()` scales the features.
- `SVC()` creates the Support Vector Classifier.
- `kernel="rbf"` uses the Radial Basis Function kernel.
- `C=1.0` controls regularization.
- `gamma="scale"` automatically calculates gamma based on the training data.

### Step 5: Train Model

```python
model.fit(X_train, y_train)
```

The model learns the decision boundaries from the training data.

### Step 6: Make Predictions

```python
y_pred = model.predict(X_test)

print(y_pred)
```

### Step 7: Evaluate Model

```python
print("Accuracy:", accuracy_score(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Classification Report:")
print(classification_report(y_test, y_pred))
```

### Step 8: Predict a New Sample

```python
new_flower = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(new_flower)

print("Predicted Species:", data.target_names[prediction[0]])
```

The pipeline automatically scales the new sample before prediction.

## 8. Advantages of SVM

- Effective in high-dimensional feature spaces.
- Works well when there is a clear margin of separation.
- Kernel trick handles nonlinear patterns.
- Uses only support vectors to define the decision boundary.
- Can work well with relatively small and medium-sized datasets.

## 9. Disadvantages of SVM

- Training can be computationally expensive on large datasets.
- Choosing the right kernel and hyperparameters can be difficult.
- Sensitive to feature scaling.
- Less interpretable than simple linear models.
- Probability estimation is not enabled by default in `SVC`.

## 10. SVM vs Logistic Regression

| Logistic Regression | SVM |
|---|---|
| Models class probabilities | Finds a maximum-margin boundary |
| Uses sigmoid for binary classification | Uses a decision function |
| Common loss: Log Loss | Common objective: Hinge loss with regularization |
| Naturally provides probabilities | Probability estimates require additional calibration |
| Linear by default | Linear or nonlinear using kernels |

## 11. Common Interview Questions

**Q1. What is the main objective of SVM?**

To find a decision boundary that maximizes the margin between different classes.

**Q2. What are support vectors?**

Data points closest to the decision boundary that determine the optimal hyperplane.

**Q3. What is the kernel trick?**

It allows SVM to learn nonlinear boundaries by implicitly computing inner products in a higher-dimensional feature space.

**Q4. What happens when C increases?**

The model penalizes classification errors and margin violations more heavily. It may fit the training data more closely.

**Q5. What happens when gamma increases in RBF SVM?**

Each training point has a smaller influence region, potentially creating a more complex decision boundary.

**Q6. Why is feature scaling important in SVM?**

Because SVM uses distances and dot products. Features with large numerical ranges can dominate the calculations.

**Q7. Can SVM perform regression?**

Yes. Support Vector Regression (SVR) is used for regression tasks.

**Q8. Can SVM handle multiclass classification?**

Yes. Scikit-learn's SVC uses One-vs-One by default for multiclass classification.

**Q9. What is the difference between hard margin and soft margin?**

- Hard margin: Does not allow margin violations.
- Soft margin: Allows violations using slack variables.

**Q10. Does SVM directly provide probabilities?**

No. `SVC` provides decision scores by default. Probability estimates can be enabled using `probability=True`, which involves additional probability calibration.

## 12. Quick Revision

- SVM = Supervised Learning.
- Mainly used for classification; SVR for regression.
- Hyperplane = Decision boundary.
- Support vectors = Closest influential data points.
- Margin = Distance between class boundaries.
- Main objective = Maximize margin.
- Kernel trick = Handle nonlinear patterns.
- C = Regularization parameter.
- Gamma = Controls influence range in RBF kernel.
- Feature scaling = Important.
- Scikit-learn classifier = `SVC()`.

## 13. Preparation Priority

| Priority | Topics |
|---|---|
| Must Know | Hyperplane, support vectors, margin |
| Must Know | Linear vs nonlinear SVM, kernel trick |
| Must Know | C, gamma, implementation |
| Important | Hard margin vs soft margin |
| Important | Hinge loss, SVR |
| Advanced | Dual optimization, Lagrange multipliers, mathematical derivation |