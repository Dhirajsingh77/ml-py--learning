# K-Nearest Neighbors (KNN) Algorithm

## 1. What is KNN?

**K-Nearest Neighbors (KNN)** is a supervised Machine Learning algorithm used for both classification and regression.

It predicts the output of a new data point by looking at the K closest data points in the training dataset.

- K represents the number of nearest neighbors considered.
- It uses distance measures to identify nearby data points.
- It is a non-parametric and instance-based learning algorithm.

**Example:**

Suppose we want to classify a fruit as an Apple or Orange.

KNN checks the K nearest fruits based on features such as weight and size, then predicts the class based on their neighbors.

## 2. How Does KNN Work?

Steps:

1. Choose the value of K.
2. Calculate the distance between the new data point and all training data points.
3. Sort the distances in ascending order.
4. Select the K nearest neighbors.
5. For classification, use majority voting.
6. For regression, calculate the average (or weighted average) of the neighbors' target values.
7. Predict the output.

## 3. Example of KNN Classification

Suppose we have the following dataset:

| Point | Weight | Size | Class |
|---|---:|---:|---|
| A | 150 | 7 | Apple |
| B | 160 | 8 | Apple |
| C | 170 | 9 | Apple |
| D | 200 | 12 | Orange |
| E | 210 | 13 | Orange |

New fruit: Weight = 165, Size = 8.

Let K = 3.

Suppose the three nearest neighbors are:

- A → Apple
- B → Apple
- C → Apple

Majority vote = Apple.

**Prediction: Apple**

## 4. Distance Metrics in KNN

KNN uses distance metrics to measure how close two data points are.

### A. Euclidean Distance

The most commonly used distance metric.

For two points in 2D:

\[
d=\sqrt{(x_2-x_1)^2+(y_2-y_1)^2}
\]

For n dimensions:

\[
d=\sqrt{\sum_{i=1}^{n}(x_i-y_i)^2}
\]

### B. Manhattan Distance

Calculates the sum of absolute differences between coordinates.

\[
d=\sum_{i=1}^{n}|x_i-y_i|
\]

### C. Minkowski Distance

A generalized distance metric.

\[
d=\left(\sum_{i=1}^{n}|x_i-y_i|^p\right)^{1/p}
\]

- p = 1: Manhattan distance.
- p = 2: Euclidean distance.

## 5. Choosing the Value of K

The value of K plays an important role in model performance.

- Small K: Sensitive to noise and may cause overfitting.
- Large K: May oversimplify patterns and cause underfitting.

**Example:**
- K = 1: Prediction depends on the closest neighbor.
- K = 3: Prediction depends on the three closest neighbors.
- K = 5: Prediction depends on the five closest neighbors.

### How to choose the best K?

- Try different K values.
- Evaluate each using validation data or cross-validation.
- Select the K that gives the best validation performance.

**Note:** An odd K is often used in binary classification to reduce the chance of ties, but it is not a universal rule.

## 6. Important Characteristics of KNN

- **Supervised Learning:** Requires labeled training data.
- **Non-parametric:** Does not assume a fixed mathematical form for the underlying data distribution.
- **Instance-based:** Stores training examples and uses them during prediction.
- **Lazy Learning:** Does not build an explicit model during training; most computation happens during prediction.

## 7. Feature Scaling in KNN

Feature scaling is very important in KNN because it uses distance calculations.

Example:

- Age: 18–60
- Salary: 20,000–1,00,000

Salary has a much larger numerical range and may dominate the distance calculation.

**Solution:** Apply feature scaling using:
- StandardScaler
- MinMaxScaler

## 8. Advantages of KNN

- Simple and easy to understand.
- Easy to implement.
- No explicit model training required.
- Can be used for classification and regression.
- Can model complex decision boundaries.

## 9. Disadvantages of KNN

- Prediction can be computationally expensive for large datasets.
- Requires more memory to store training data.
- Sensitive to feature scaling.
- Sensitive to irrelevant features and noise.
- Performance can degrade in high-dimensional data (curse of dimensionality).

## 10. KNN Implementation Using Python

### Step 1: Import Libraries

```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
```

### Step 2: Create Dataset

```python
X = np.array([
    [150, 7],
    [160, 8],
    [170, 9],
    [200, 12],
    [210, 13],
    [220, 14]
])

y = np.array([
    "Apple",
    "Apple",
    "Apple",
    "Orange",
    "Orange",
    "Orange"
])
```

- X = Input features (weight and size).
- y = Target labels (fruit classes).

### Step 3: Split Dataset

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.33,
    random_state=42,
    stratify=y
)
```

### Step 4: Feature Scaling

```python
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### Step 5: Create and Train KNN Model

```python
model = KNeighborsClassifier(n_neighbors=3)

model.fit(X_train, y_train)
```

### Step 6: Make Predictions

```python
y_pred = model.predict(X_test)

print("Predictions:", y_pred)
print("Actual:", y_test)
```

### Step 7: Evaluate Model

```python
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
```

### Step 8: Predict a New Data Point

```python
new_fruit = np.array([[165, 8]])

new_fruit_scaled = scaler.transform(new_fruit)

prediction = model.predict(new_fruit_scaled)

print("Predicted Fruit:", prediction[0])
```

**Important Scikit-learn functions:**

| Function | Purpose |
|---|---|
| `KNeighborsClassifier()` | Creates KNN classifier |
| `fit()` | Stores training data |
| `predict()` | Predicts class labels |
| `predict_proba()` | Returns class probabilities |
| `score()` | Returns mean accuracy for classification |
| `StandardScaler()` | Standardizes features |

## 11. Time and Space Complexity

Let:
- n = Number of training samples.
- d = Number of features.

| Operation | Complexity |
|---|---|
| Training / fitting | Approximately O(1) for basic brute-force KNN |
| One prediction | O(nd) |
| Predicting m samples | O(mnd) |
| Storage | O(nd) |

*Note: Actual performance depends on the search algorithm. KD-trees and Ball Trees can accelerate some low-dimensional queries.*

## 12. Important Interview Questions

**Q1. What is KNN?**

A supervised, non-parametric, instance-based algorithm that predicts using the K nearest training examples.

**Q2. Why is KNN called a lazy learner?**

Because it does not build an explicit predictive model during training. It performs most calculations when making predictions.

**Q3. Why is feature scaling important in KNN?**

Because KNN uses distance calculations, and features with larger numerical ranges can dominate the distance.

**Q4. What happens when K = 1?**

The prediction depends on the single nearest neighbor. It can be sensitive to noise and overfitting.

**Q5. What happens when K is too large?**

The model may become too generalized and underfit the data.

**Q6. Can KNN be used for regression?**

Yes. It predicts a numerical output using the average or weighted average of the nearest neighbors' target values.

**Q7. What is the curse of dimensionality?**

As the number of features increases, distance measurements become less informative and nearest-neighbor methods can become less effective.

**Q8. What is the difference between KNN and K-Means?**

- KNN: Supervised learning; predicts using labeled neighbors.
- K-Means: Unsupervised learning; groups data into clusters.

**Q9. Is KNN a parametric or non-parametric algorithm?**

KNN is non-parametric because it does not assume a fixed functional form for the relationship between input and output.

**Q10. How do you select the optimal K?**

Use validation data or cross-validation to compare different K values and select the best-performing one.

## 13. Quick Revision

| Concept | Remember |
|---|---|
| KNN | K-Nearest Neighbors |
| Learning Type | Supervised |
| Applications | Classification and Regression |
| K | Number of neighbors |
| Distance | Euclidean, Manhattan, Minkowski |
| Classification | Majority voting |
| Regression | Average of neighbors |
| Learning Type | Lazy learning |
| Scaling | Very important |
| Small K | Risk of overfitting |
| Large K | Risk of underfitting |
| Main Limitation | Expensive prediction on large datasets |

