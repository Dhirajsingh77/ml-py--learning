# Training a Machine Learning Model

## 1. What is Model Training?

Model training is the process of teaching a Machine Learning algorithm to learn patterns and relationships from training data so that it can make predictions on unseen data.

**How it works:**
1. The model receives input features (X).
2. It predicts an output (\(\hat{Y}\)).
3. It compares predictions with actual outputs (Y).
4. It calculates the error using a loss function.
5. It updates its parameters to reduce the error.
6. This process repeats until the model learns an appropriate relationship.

## 2. Steps Involved in Training a Model

**Machine Learning Workflow:**

1. **Data Collection:** Collect relevant datasets.
2. **Data Preprocessing:** Handle missing values, duplicates and categorical data.
3. **Feature Selection:** Choose relevant input features.
4. **Dataset Splitting:** Divide data into training, validation and testing sets.
5. **Model Selection:** Choose a suitable ML algorithm.
6. **Model Training:** Learn patterns and update model parameters.
7. **Model Evaluation:** Measure performance using evaluation metrics.
8. **Prediction / Deployment:** Use the trained model on new data.

## 3. Dataset Splitting

A dataset is generally divided into three parts:

| Dataset | Purpose | Typical Split |
|---|---|---:|
| Training Set | Used to train the model | 70% |
| Validation Set | Used for tuning hyperparameters | 15% |
| Testing Set | Used for final evaluation | 15% |

**Note:** These percentages are common examples, not fixed rules.

### Important Terms

- **Training Data:** Data used by the model to learn.
- **Validation Data:** Data used to tune the model and select the best configuration.
- **Testing Data:** Independent data used to evaluate final model performance.

## 4. How Does a Model Learn?

A model learns by adjusting its parameters to minimize prediction error.

**Example: Predicting House Prices using Linear Regression**

- Input: House area = 1000 sq. ft.
- Actual price (Y) = ₹50 lakh.
- Model prediction (\(\hat{Y}\)) = ₹40 lakh.
- Error = ₹10 lakh.

The model adjusts its parameters to reduce the prediction error.

This process continues over multiple training examples.

## 5. Important Training Concepts

### A. Parameters

- Values learned automatically by the model during training.
- Example: Weights and bias in Linear Regression.

### B. Hyperparameters

- Settings specified before or during training that control the learning process.
- They are not directly learned from the training data.

**Examples:**
- Learning rate
- Number of epochs
- Batch size
- Maximum tree depth
- Number of neighbors (K in KNN)

### C. Loss Function

- Measures the difference between predicted and actual outputs.
- The training process attempts to minimize this loss.

**Example: Mean Squared Error (MSE)**

\[
MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
\]

Where:
- \(n\) = Number of samples
- \(y_i\) = Actual value
- \(\hat{y}_i\) = Predicted value

### D. Epoch

- One complete pass through the entire training dataset.

### E. Batch Size

- Number of training samples processed before the model updates its parameters.

### F. Learning Rate

- Controls how large a step the model takes when updating its parameters.

## 6. Training vs Testing

| Training | Testing |
|---|---|
| Model learns patterns | Model is evaluated |
| Uses training dataset | Uses testing dataset |
| Updates parameters | Does not update parameters |
| Minimizes training loss | Measures performance on unseen data |

## 7. Important Interview Questions

**Q1. What is model training?**

The process of learning patterns from training data by adjusting model parameters to minimize prediction error.

**Q2. Difference between parameters and hyperparameters?**

- Parameters are learned from data.
- Hyperparameters are settings chosen before or during training.

**Q3. What is an epoch?**

One complete pass through the entire training dataset.

**Q4. Why do we split the dataset?**

To train the model, tune its configuration and independently evaluate its ability to generalize to unseen data.

**Q5. What is a loss function?**

A mathematical function that measures the error between actual and predicted outputs.

**Q6. What is the learning rate?**

It controls the step size used to update model parameters during optimization.

**Q7. Can we use testing data during training?**

No. Testing data should remain separate and be used only for final evaluation to avoid data leakage.

## 8. Quick Revision

| Term | Meaning |
|---|---|
| Model Training | Learning patterns from data |
| Training Set | Used to train |
| Validation Set | Used for tuning |
| Testing Set | Used for final evaluation |
| Parameters | Learned by the model |
| Hyperparameters | Configured by the developer |
| Loss Function | Measures prediction error |
| Epoch | One complete pass over training data |
| Batch Size | Samples processed per update |
| Learning Rate | Controls update step size |
