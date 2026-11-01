
# Introduction to Machine Learning (ML)

## 1. Definition of Machine Learning

**Machine Learning (ML)** is a subset of Artificial Intelligence (AI) that enables computers to learn from data and improve their performance without being explicitly programmed for every task.

- ML identifies patterns in data.
- Uses these patterns to make predictions or decisions.
- Performance improves with experience and data.

**Example:**
- Email spam detection.
- Movie recommendations.
- House price prediction.
- Face recognition.

## 2. Difference Between AI vs ML vs Data Science

### Artificial Intelligence (AI)
- AI is a branch of computer science that focuses on creating machines capable of performing tasks that typically require human intelligence.
- Examples: Chatbots, self-driving cars, virtual assistants.

### Machine Learning (ML)
- ML is a subset of AI that enables machines to learn patterns from data and make predictions without explicitly programming every rule.
- Examples: Spam detection, recommendation systems, price prediction.

### Data Science (DS)
- Data Science is an interdisciplinary field that uses statistics, programming, mathematics, and domain knowledge to extract meaningful insights from data.
- Examples: Sales analysis, customer behavior analysis, business forecasting.

## 3. AI vs ML vs Data Science

| Basis | AI | ML | Data Science |
|---|---|---|---|
| Full Form | Artificial Intelligence | Machine Learning | Data Science |
| Purpose | Build intelligent systems | Learn from data | Extract insights from data |
| Focus | Intelligence and decision-making | Learning and prediction | Data analysis and interpretation |
| Techniques | Search, reasoning, ML, etc. | Algorithms and statistical models | Statistics, visualization, ML |
| Output | Intelligent behavior | Predictions or decisions | Insights and conclusions |
| Example | Chatbot | Spam detection | Sales analysis |

## 4. Relationship Between AI, ML and Data Science

- ML is a subset of AI.
- Data Science is a separate but overlapping field.
- Data Science can use ML techniques to analyze and predict outcomes.
- AI can use ML to develop intelligent systems.

**Remember:**

AI ⊃ ML

Data Science overlaps with AI and ML but is not a subset of either.

## 5. Real-World Example

Consider an online shopping platform:

- **AI:** An intelligent shopping assistant that interacts with customers.
- **ML:** Recommends products based on customer purchase history.
- **Data Science:** Analyzes customer behavior, sales trends, and purchasing patterns.

## 6. Important Interview Questions

**Q1. What is Machine Learning?**

ML is a subset of AI that enables computers to learn from data and improve performance without explicitly programming every rule.

**Q2. Is ML a subset of AI?**

Yes. ML is a subset of AI.

**Q3. Is Data Science a subset of ML?**

No. Data Science is an independent interdisciplinary field that overlaps with ML.

**Q4. Can AI exist without ML?**

Yes. AI can use rule-based systems, search algorithms, and logical reasoning without ML.

**Q5. What is the main difference between AI and ML?**

AI focuses on creating intelligent systems, whereas ML focuses on enabling systems to learn from data.

---

## Quick Revision

- **AI:** Making machines intelligent.
- **ML:** Making machines learn from data.
- **DS:** Extracting useful insights from data.

**Interview Tip:** Remember that ML is a subset of AI, while Data Science is an overlapping interdisciplinary field.

# Types of Machine Learning

Machine Learning is mainly classified into **4 types** based on how a model learns from data.

1. Supervised Learning
2. Unsupervised Learning
3. Semi-Supervised Learning
4. Reinforcement Learning

## 1. Supervised Learning

- Supervised Learning is a type of ML in which a model learns from **labeled data**.
- Labeled data means both input (features) and correct output (target) are provided.
- The model learns the relationship between input and output to predict results for new data.

**Example:**

Predicting house prices using area, location, and number of bedrooms.

| Area (sq. ft.) | Bedrooms (X) | Price (Y) |
|---|---|---|
| 1200 | 2 | 40 lakh |
| 1800 | 3 | 70 lakh |
| 1500 | 2 | 50 lakh |

Here, price is the target variable.

### Types of Supervised Learning

**A. Regression**
- Used to predict continuous numerical values.
- Output is a number.

Examples:
- House price prediction.
- Temperature prediction.
- Salary prediction.

Algorithms:
- Linear Regression
- Decision Tree Regression
- Random Forest Regression

**B. Classification**
- Used to predict discrete categories or classes.
- Output is a class label.

Examples:
- Spam or Not Spam.
- Pass or Fail.
- Disease detection.

Algorithms:
- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Naive Bayes

## 2. Unsupervised Learning

- Unsupervised Learning is a type of ML in which a model learns from **unlabeled data**.
- No predefined output or target variable is provided.
- The model discovers hidden patterns, structures, and relationships in data.

**Example:**

An e-commerce company groups customers based on their shopping behavior without predefined customer categories.

### Types of Unsupervised Learning

**A. Clustering**
- Groups similar data points into clusters.
- Data points within the same cluster are more similar to each other.

Examples:
- Customer segmentation.
- Grouping similar documents.

Algorithms:
- K-Means
- Hierarchical Clustering
- DBSCAN

**B. Association**
- Finds relationships and frequently occurring patterns between items.

Example:
- Customers who buy bread often buy butter.

Algorithms:
- Apriori
- FP-Growth

**C. Dimensionality Reduction**
- Reduces the number of features while preserving important information.

Example:
- Reducing 100 features to 10 important features.

Algorithms:
- PCA (Principal Component Analysis)
- t-SNE

## 3. Semi-Supervised Learning

- Uses a combination of **labeled and unlabeled data**.
- Usually, a small amount of labeled data and a large amount of unlabeled data are available.
- Helps train models when labeling data is expensive or time-consuming.

**Example:**
- Image classification where only a few images are labeled, but thousands of images are unlabeled.
- Speech recognition.

## 4. Reinforcement Learning

- A type of ML in which an **agent learns by interacting with an environment**.
- The agent performs actions and receives rewards or penalties.
- The goal is to maximize the total cumulative reward over time.

### Important Components

| Component | Meaning |
|---|---|
| Agent | Learner or decision-maker |
| Environment | Surroundings in which the agent operates |
| Action | Decision taken by the agent |
| Reward | Feedback received after an action |
| Policy | Strategy used to choose actions |

**Example:**
- Training a robot to walk.
- Self-driving cars.
- Game-playing AI.

### Working of Reinforcement Learning

1. Agent observes the environment.
2. Agent takes an action.
3. Environment provides a reward or penalty.
4. Agent learns from feedback.
5. Process repeats to improve its strategy.

## 5. Comparison of All Types

| Type | Training Data | Learning Method | Example |
|---|---|---|---|
| Supervised | Labeled | Learns input-output mapping | Spam detection |
| Unsupervised | Unlabeled | Finds hidden patterns | Customer clustering |
| Semi-Supervised | Both | Learns from limited labels and more unlabeled data | Image classification |
| Reinforcement | Rewards and interactions | Trial and error | Game-playing AI |

## 6. Important Interview Questions

**Q1. What are the main types of Machine Learning?**

- Supervised Learning
- Unsupervised Learning
- Semi-Supervised Learning
- Reinforcement Learning

**Q2. What is the difference between supervised and unsupervised learning?**

- Supervised: Uses labeled data.
- Unsupervised: Uses unlabeled data.

**Q3. What is the difference between classification and regression?**

- Classification predicts categories.
- Regression predicts continuous numerical values.

**Q4. What is clustering?**

Clustering is an unsupervised learning technique that groups similar data points together.

**Q5. What is Reinforcement Learning?**

It is a learning technique in which an agent learns through interaction with an environment using rewards and penalties.

**Q6. Why is Semi-Supervised Learning useful?**

It is useful when labeled data is limited or expensive to obtain, but a large amount of unlabeled data is available.

---


## 1. What is a Feature?

- A **feature** is an individual measurable property or attribute of a dataset used by an ML model to make predictions.
- Also known as input variable, attribute, or independent variable.
- Features help models identify patterns in data.

**Example: House Price Prediction**

| Area | Bedrooms | Location | Price |
|---|---|---|---|
| 1200 sq.ft. | 2 | Kolkata | 40 lakh |
| 1800 sq.ft. | 3 | Delhi | 70 lakh |

- Features (X): Area, Bedrooms, Location.
- Target (Y): Price.

## 2. Types of Features

### A. Quantitative (Numerical) Features
Represented using numbers.

1. **Continuous:** Can take any value within a range.
   - Examples: Height, Weight, Temperature.
2. **Discrete:** Takes countable values.
   - Examples: Number of students, Bedrooms, Cars.

### B. Qualitative (Categorical) Features
Represent categories or groups.

1. **Nominal:** Categories without any natural order.
   - Examples: Color, City, Blood Group.
2. **Ordinal:** Categories with a meaningful order.
   - Examples: Low, Medium, High; Education Level.

## 3. Feature vs Target Variable

| Feature (X) | Target (Y) |
|---|---|
| Independent variable | Dependent variable |
| Input to the model | Output to be predicted |
| Used for prediction | Prediction objective |

Example: Predicting marks using study hours.
- X = Study Hours
- Y = Exam Marks

## 4. Feature Engineering

Feature engineering is the process of creating, transforming, and selecting features to improve model performance.

**Common techniques:**

- **Feature Creation:** Creating new features from existing data.
- **Feature Transformation:** Changing the representation of features.
- **Feature Selection:** Selecting relevant features.
- **Feature Scaling:** Bringing numerical features to a comparable scale.
- **Feature Encoding:** Converting categorical features into numerical form.

## 5. Feature Encoding

ML models generally require numerical input.

| Technique | Description |
|---|---|
| Label Encoding | Assigns numerical labels to categories. |
| One-Hot Encoding | Creates separate binary columns for categories. |
| Ordinal Encoding | Assigns numbers based on category order. |

**Example: One-Hot Encoding**

Original data:

| Color |
|---|
| Red |
| Blue |
| Green |

Encoded data:

| Red | Blue | Green |
|---|---|---|
| 1 | 0 | 0 |
| 0 | 1 | 0 |
| 0 | 0 | 1 |

## 6. Feature Scaling

Feature scaling brings numerical features to a comparable scale.

### A. Normalization (Min-Max Scaling)

Scales values, generally between 0 and 1.

Formula:

X' = (X - X_min) / (X_max - X_min)

### B. Standardization (Z-score)

Transforms data to have:
- Mean = 0
- Standard Deviation = 1

Formula:

Z = (X - μ) / σ

Where:
- X = Original value
- μ = Mean
- σ = Standard deviation

**Why is scaling important?**
- Prevents large-range features from dominating certain algorithms.
- Helps distance-based algorithms such as KNN and SVM.
- Can improve gradient descent convergence.

## 7. Important Interview Questions

**Q1. What is a feature in ML?**
- An input attribute used by a model to make predictions.

**Q2. Difference between feature and target?**
- Feature (X): Input variable.
- Target (Y): Output variable to predict.

**Q3. What is feature engineering?**
- Creating, transforming, and selecting features to improve model performance.

**Q4. Difference between nominal and ordinal features?**
- Nominal: No natural order (e.g., colors).
- Ordinal: Meaningful order (e.g., Low, Medium, High).

**Q5. Why is feature scaling important?**
- It prevents features with larger numerical ranges from dominating certain algorithms.

**Q6. Difference between normalization and standardization?**
- Normalization: Usually scales values between 0 and 1.
- Standardization: Transforms values using mean and standard deviation.

---


