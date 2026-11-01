# Classification vs Regression in Machine Learning

Both **Classification and Regression** are types of Supervised Learning because they use labeled data to train a model.

## 1. Classification

### Definition
Classification is a supervised learning technique used to predict **categorical or discrete class labels** based on input features.

- Output is a category or class.
- Learns from labeled training data.
- Used when the prediction belongs to a predefined class.

### Examples
- Email: Spam or Not Spam.
- Student: Pass or Fail.
- Disease: Positive or Negative.
- Image: Cat, Dog or Bird.

### Types of Classification

1. **Binary Classification**
   - Contains only two possible classes.
   - Example: Spam / Not Spam.

2. **Multiclass Classification**
   - Contains more than two mutually exclusive classes.
   - Example: Classifying an image as Cat, Dog or Bird.

3. **Multilabel Classification**
   - A single data point can belong to multiple classes simultaneously.
   - Example: An image containing a Cat, Dog and Car.
   
  ** Rule of thumb: If picking one label forces the others to be false → multiclass. If labels can independently be true or false → multilabel.

### Common Classification Algorithms
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Naive Bayes

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## 2. Regression

### Definition
Regression is a supervised learning technique used to predict **continuous numerical values** based on input features.

- Output is a numerical value.
- Learns the relationship between input and output.
- Used when the prediction is a quantity.

### Examples
- House price prediction.
- Salary prediction.
- Temperature prediction.
- Sales forecasting.

### Types of Regression

1. **Simple Linear Regression**
   - Uses one independent variable to predict one dependent variable.
   - Example: Predicting salary using years of experience.

2. **Multiple Linear Regression**
   - Uses two or more independent variables.
   - Example: Predicting house price using area, bedrooms and location.

3. **Polynomial Regression**
   - Models nonlinear relationships using polynomial terms.
   - Example: Predicting growth trends.

### Linear Regression Formula

\[
y = mx + c
\]

Where:
- \(y\) = Predicted output.
- \(x\) = Input feature.
- \(m\) = Slope.
- \(c\) = Intercept.

For multiple linear regression:

\[
y = b_0+b_1x_1+b_2x_2+\cdots+b_nx_n
\]

### Common Regression Algorithms
- Linear Regression
- Polynomial Regression
- Decision Tree Regression
- Random Forest Regression
- Support Vector Regression (SVR)

### Evaluation Metrics
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- R² Score (Coefficient of Determination)

## 3. Classification vs Regression

| Basis | Classification | Regression |
|---|---|---|
| Output | Categorical | Continuous numerical |
| Purpose | Predict class labels | Predict numerical values |
| Data type | Discrete classes | Continuous values |
| Example | Spam / Not Spam | House price |
| Algorithms | Logistic Regression, KNN, SVM | Linear Regression, SVR |
| Evaluation | Accuracy, Precision, Recall | MAE, MSE, RMSE, R² |

## 4. Real-World Example

Consider a student performance prediction system.

**Classification:**
- Input: Study hours, attendance.
- Output: Pass or Fail.

**Regression:**
- Input: Study hours, attendance.
- Output: Expected marks (e.g., 85.5).

## 5. Important Interview Questions

**Q1. What is the main difference between classification and regression?**

- Classification predicts categorical outputs.
- Regression predicts continuous numerical outputs.

**Q2. Is Logistic Regression used for classification or regression?**

Despite its name, Logistic Regression is primarily used for classification.

**Q3. Can classification have more than two classes?**

Yes. It can perform multiclass and multilabel classification.

**Q4. What is the difference between binary and multiclass classification?**

- Binary: Two classes.
- Multiclass: More than two mutually exclusive classes.

**Q5. What is the difference between MAE and MSE?**

- MAE: Average absolute prediction error.
- MSE: Average squared prediction error.

**Q6. What is R² score?**

It measures how much variation in the target variable is explained by the regression model. A higher value generally indicates a better fit.

---

## 6. Quick Revision

| Concept | Remember |
|---|---|
| Classification | Predicts categories |
| Regression | Predicts numerical values |
| Binary Classification | Two classes |
| Multiclass Classification | Multiple mutually exclusive classes |
| Multilabel Classification | Multiple labels per sample |
| Linear Regression | Fits a straight-line relationship |
| Logistic Regression | Classification algorithm |
| Accuracy | Classification metric |
| MAE / MSE / RMSE | Regression metrics |
| R² | Goodness of fit |
