# Naive Bayes Algorithm in Machine Learning

## 1. What is Naive Bayes?

**Naive Bayes** is a supervised Machine Learning algorithm based on Bayes' Theorem, mainly used for classification problems.

It predicts the probability of a data point belonging to different classes and selects the class with the highest probability.

**Why is it called Naive Bayes?**
- **Bayes:** Based on Bayes' Theorem.
- **Naive:** Assumes that features are conditionally independent given the class label.

**Applications:**
- Email spam detection.
- Sentiment analysis.
- Document classification.
- News categorization.
- Medical diagnosis.

## 2. Bayes' Theorem

Bayes' Theorem calculates the probability of an event based on previously known information.

### Formula

\[
P(A|B)=\frac{P(B|A)\times P(A)}{P(B)}
\]

Where:

- P(A|B) = Posterior probability.
- P(B|A) = Likelihood.
- P(A) = Prior probability.
- P(B) = Evidence.

### Understanding the Terms

| Term | Meaning |
|---|---|
| Prior | Initial probability of a class |
| Likelihood | Probability of observing the features given a class |
| Evidence | Overall probability of observing the features |
| Posterior | Updated probability of a class after observing features |

## 3. How Does Naive Bayes Work?

Suppose an email application needs to classify an email as Spam or Not Spam.

Steps:

1. Collect labeled emails (Spam and Not Spam).
2. Calculate the prior probability of each class.
3. Calculate the likelihood of words appearing in each class.
4. Use Bayes' Theorem to calculate posterior probabilities.
5. Compare the probabilities of different classes.
6. Assign the email to the class with the highest probability.

### Example

Suppose an email contains the word "Offer".

Assume:

- P(Spam) = 0.4
- P(Not Spam) = 0.6
- P(Offer | Spam) = 0.8
- P(Offer | Not Spam) = 0.1

Using Bayes' Theorem:

\[
P(Spam|Offer)=
\frac{0.8\times0.4}
{P(Offer)}
\]

First, calculate evidence:

\[
P(Offer)=
(0.8\times0.4)+(0.1\times0.6)
\]

\[
P(Offer)=0.38
\]

Therefore:

\[
P(Spam|Offer)=\frac{0.32}{0.38}
\]

\[
P(Spam|Offer)=0.8421
\]

\[
P(Spam|Offer)=84.21\%
\]

Similarly:

\[
P(NotSpam|Offer)=15.79\%
\]

**Prediction: Spam**

## 4. Naive Bayes Mathematical Formula

For multiple features:

\[
P(C|X_1,\ldots,X_n)
\propto P(C)\prod_{i=1}^{n}P(X_i|C)
\]

Where:

- C = Class.
- X₁, X₂, ..., Xₙ = Input features.
- P(C) = Prior probability.
- P(Xᵢ|C) = Conditional probability of a feature given the class.

The model predicts the class with the highest posterior probability.

**Important:** The independence assumption allows the joint likelihood to be calculated as a product of individual feature likelihoods.

## 5. Types of Naive Bayes

### A. Gaussian Naive Bayes

- Used for continuous numerical features.
- Assumes that feature values follow a Gaussian (normal) distribution within each class.

Examples:
- Predicting disease using temperature and blood pressure.
- Classifying data using height and weight.

### B. Multinomial Naive Bayes

- Used for discrete feature counts or frequencies.
- Commonly used in text classification.

Examples:
- Spam detection.
- News classification.
- Document categorization.

### C. Bernoulli Naive Bayes

- Used for binary-valued features.
- Features represent whether something is present (1) or absent (0).

Examples:
- Whether a word appears in an email.
- Binary text classification.

### Comparison

| Type | Data | Example |
|---|---|---|
| Gaussian NB | Continuous numerical | Medical measurements |
| Multinomial NB | Counts/frequencies | Word counts in emails |
| Bernoulli NB | Binary features | Word present or absent |

## 6. Laplace Smoothing

**Problem: Zero Probability**

Suppose a word never appeared in the Spam training emails.

Its probability becomes zero.

Since Naive Bayes multiplies probabilities, one zero probability can make the entire class likelihood zero.

**Solution: Laplace Smoothing**

Laplace smoothing adds a small count (usually 1) to avoid zero probabilities.

### Formula

\[
P(w|C)=\frac{Count(w,C)+\alpha}
{TotalWords(C)+\alpha V}
\]

Where:

- Count(w,C) = Frequency of word w in class C.
- α = Smoothing parameter (usually 1).
- V = Vocabulary size.
- TotalWords(C) = Total word occurrences in class C.

**Interview Point:** Laplace smoothing prevents zero probabilities for unseen features.

## 7. Advantages of Naive Bayes

- Simple and easy to implement.
- Fast training and prediction.
- Works well with high-dimensional data.
- Effective for text classification.
- Requires relatively little training data.
- Supports multiclass classification.

## 8. Disadvantages of Naive Bayes

- Assumes conditional independence between features.
- Performance may decrease when features are highly dependent.
- Zero-frequency problems require smoothing.
- Probability estimates may not be well calibrated.
- Gaussian Naive Bayes depends on its distribution assumption.

## 9. Naive Bayes Implementation Using Python

### Example: Email Spam Classification

We will use Multinomial Naive Bayes with a text dataset.

### Step 1: Import Libraries

```python
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
```

### Step 2: Create Dataset

```python
data = {
    "message": [
        "Win money now",
        "Claim your free prize",
        "Meeting scheduled tomorrow",
        "Let's have lunch",
        "Free offer click now",
        "Project meeting today",
        "Win a free gift",
        "Please send the report"
    ],

    "label": [
        "spam",
        "spam",
        "ham",
        "ham",
        "spam",
        "ham",
        "spam",
        "ham"
    ]
}

df = pd.DataFrame(data)
```

### Step 3: Separate Features and Target

```python
X = df["message"]
y = df["label"]
```

- X = Email messages.
- y = Spam or Ham labels.

### Step 4: Split Dataset

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)
```

### Step 5: Convert Text into Numerical Features

ML algorithms require numerical input.

We use CountVectorizer to convert text into word-count vectors.

```python
vectorizer = CountVectorizer()

X_train = vectorizer.fit_transform(X_train)

X_test = vectorizer.transform(X_test)
```

**Important:**
- `fit_transform()` learns the vocabulary and converts training text.
- `transform()` converts test text using the same vocabulary.

### Step 6: Create and Train Model

```python
model = MultinomialNB()

model.fit(X_train, y_train)
```

### Step 7: Make Predictions

```python
y_pred = model.predict(X_test)

print(y_pred)
```

### Step 8: Evaluate Model

```python
print("Accuracy:", accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))
```

### Step 9: Predict a New Email

```python
new_email = ["Congratulations! You won a free prize"]

new_email_vector = vectorizer.transform(new_email)

prediction = model.predict(new_email_vector)

print("Prediction:", prediction[0])
```

**Note:** This is a tiny demonstration dataset, not a reliable real-world spam classifier. Its evaluation metrics should not be treated as representative performance.

## 10. Important Scikit-learn Functions

| Function | Purpose |
|---|---|
| `MultinomialNB()` | Creates Multinomial Naive Bayes |
| `GaussianNB()` | Creates Gaussian Naive Bayes |
| `BernoulliNB()` | Creates Bernoulli Naive Bayes |
| `fit()` | Trains the model |
| `predict()` | Predicts class labels |
| `predict_proba()` | Returns predicted class probabilities |
| `CountVectorizer()` | Converts text into word-count vectors |
| `TfidfVectorizer()` | Converts text into TF-IDF features |

## 11. Time and Space Complexity

Let:
- N = Number of training samples.
- D = Number of features.
- C = Number of classes.

For dense feature data:

| Operation | Approximate Complexity |
|---|---|
| Training | O(N × D) |
| Prediction per sample | O(C × D) |
| Storage | O(C × D) for learned statistics |

Actual complexity depends on the variant and data representation. Sparse text data can be processed more efficiently using non-zero feature counts.

## 12. Important Interview Questions

**Q1. What is Naive Bayes?**

A supervised probabilistic classification algorithm based on Bayes' Theorem and conditional independence assumptions.

**Q2. Why is it called Naive Bayes?**

Because it assumes that features are conditionally independent given the class.

**Q3. What is Bayes' Theorem?**

It calculates the posterior probability using prior probability and likelihood.

**Q4. What is the difference between prior and posterior probability?**

- Prior: Probability before observing new evidence.
- Posterior: Updated probability after observing evidence.

**Q5. What is Laplace smoothing?**

A technique that adds a small count to feature frequencies to avoid zero probabilities.

**Q6. What are the three main types of Naive Bayes?**

- Gaussian Naive Bayes.
- Multinomial Naive Bayes.
- Bernoulli Naive Bayes.

**Q7. Why is Naive Bayes useful for text classification?**

It is computationally efficient and works well with high-dimensional word-frequency features.

**Q8. Is feature independence always true in real-world datasets?**

No. It is an assumption made by the algorithm and may not hold in real-world data.

**Q9. Does Naive Bayes support multiclass classification?**

Yes. It can classify data into more than two classes.

**Q10. What is the difference between KNN and Naive Bayes?**

| KNN | Naive Bayes |
|---|---|
| Distance-based | Probability-based |
| Instance-based | Probabilistic model |
| Lazy learner | Learns class and feature statistics |
| Uses nearest neighbors | Uses Bayes' Theorem |
| Scaling is often important | Scaling is generally not required |

## 13. Quick Revision

| Concept | Remember |
|---|---|
| Naive Bayes | Probabilistic classifier |
| Learning Type | Supervised |
| Foundation | Bayes' Theorem |
| Naive Assumption | Conditional independence |
| Prior | Initial class probability |
| Likelihood | Probability of features given class |
| Posterior | Updated class probability |
| Gaussian NB | Continuous features |
| Multinomial NB | Feature counts |
| Bernoulli NB | Binary features |
| Laplace Smoothing | Prevents zero probabilities |
| Main Application | Spam detection and text classification |

## 14. Must Know vs Important vs Advanced

| Priority | Topics |
|---|---|
| Must Know | Bayes' Theorem, working, assumptions, types, implementation |
| Important | Prior, likelihood, posterior, Laplace smoothing |
| Advanced | Log probabilities, probability calibration, TF-IDF, conditional independence limitations |

