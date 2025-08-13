<h1>📊 Customer Churn Prediction</h1>

<p class="accent">Predict whether a telecom customer will churn (leave) or stay using a Random Forest model.</p>

<div class="badges">
        <li class="badge">Python 🐍</li>
        <li class="badge">scikit-learn</li>
        <li class="badge">pandas</li>
        <li class="badge">matplotlib</li>
        <li class="badge">seaborn</li>
        <li class="badge">imblearn (SMOTE)</li>
        <li class="badge ok">~86% Accuracy</li>
      </div>

<h2>📌 Overview</h2>
      <p>
        This project predicts customer <strong>churn</strong> from the Telco Customer Churn dataset using: data cleaning,
        <em>Ordinal encoding</em>, simple feature engineering, <em>SMOTE</em> to balance classes, and a tuned
        <strong>RandomForestClassifier</strong>.
      </p>

  <h2>🛠 Tech Stack</h2>
      <p>
        <span class="pill">Python 3.9+</span> ,
        <span class="pill">pandas</span> ,
        <span class="pill">numpy</span> ,
        <span class="pill">scikit-learn</span> ,
        <span class="pill">imbalanced-learn</span> ,
        <span class="pill">matplotlib</span> ,
        <span class="pill">seaborn</span>
      </p>

  <h2>📂 Dataset</h2>
      <p><strong>Target:</strong> <code>Churn</code> (Yes/No) &nbsp; • &nbsp; <strong>Key features:</strong> Contract, InternetService, MonthlyCharges, Tenure, PaymentMethod, etc.</p>

  <h2>🚀 Setup</h2>
      <pre><code>
        
# 1) Install dependencies
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
        
# 2) Place the dataset
e.g. Telco-Customer-Churn.csv in the project root

# 3) Run the script
python churn_train.py

</code></pre>

  <h2>📁 Suggested Structure</h2>
      <pre><code>.
├─ data/
│  └─ Telco-Customer-Churn.csv
├─ churn_train.py
└─ README.html
</code></pre>

  <h2>🧠 Training Script (summary)</h2>
      <div class="grid">
        <div>
          <h3>Steps</h3>
          <ul>
            <li>Load CSV</li>
            <li>Convert <code>TotalCharges</code> to numeric &amp; handle missing</li>
            <li>Ordinal-encode categorical columns</li>
            <li>Feature engineering: <code>TenurePerMonthCharges = tenure * MonthlyCharges</code></li>
            <li>SMOTE to balance classes</li>
            <li>Train/test split</li>
            <li>Train tuned RandomForest</li>
            <li>Evaluate: accuracy, classification report</li>
            <li>Plot: churn distribution, contract vs churn, MonthlyCharges histogram, feature importances</li>
          </ul>
        </div>
        <div>
          <h3>Model Params</h3>
          <ul>
            <li><code>n_estimators=200</code></li>
            <li><code>max_depth=10</code></li>
            <li><code>min_samples_split=5</code></li>
            <li><code>min_samples_leaf=3</code></li>
            <li><code>random_state=42</code></li>
          </ul>
          <p class="ok">Typical accuracy: ~86% (varies by split/random state).</p>
        </div>
      </div>

  <h1>🧩 Full Code (core pipeline)</h1>

<h2>Step 1: Import Required Libraries</h2>
<pre>
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OrdinalEncoder
from imblearn.over_sampling import SMOTE
</pre>

<h2>Step 2: Load the Dataset</h2>
<pre>
df = pd.read_csv('/content/Telco-Customer-Churn.csv')
df.sample(10)
</pre>

<h2>Step 3: Handle Missing Values and Change Data Types</h2>
<p>Convert <code>SeniorCitizen</code> to category (0 or 1)</p>
<pre>
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

categorical_cols = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod', 'Churn'
]
df[categorical_cols] = df[categorical_cols].astype('category')
df['SeniorCitizen'] = df['SeniorCitizen'].astype('category')

df.info()
</pre>

<h2>Step 4: Check Columns with NaN Values</h2>
<pre>
na_cols = df.isna().sum()
print(na_cols[na_cols > 0])
</pre>

<h2>Step 5: Remove Rows with NaN in TotalCharges</h2>
<pre>
df = df.dropna(subset=['TotalCharges'])
</pre>

<h2>Step 6: Remove customerID Column</h2>
<p>Keep a copy of <code>customerID</code> for reference.</p>
<pre>
customer_ids = df['customerID']
df.drop('customerID', axis=1, inplace=True)
</pre>

<h2>Step 7: Encode Categorical Variables</h2>
<pre>
encoder = OrdinalEncoder()
cat_cols_for_encoding = df.select_dtypes(include=['category', 'object']).columns
df[cat_cols_for_encoding] = encoder.fit_transform(df[cat_cols_for_encoding])
</pre>

<h2>Step 8: Split Features and Target</h2>
<pre>
X = df.drop('Churn', axis=1)
y = df['Churn']
</pre>

<h2>Step 9: Balance Dataset using SMOTE</h2>
<pre>
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)
</pre>

<h2>Step 10: Train-Test Split (20% Test Size)</h2>
<pre>
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
</pre>

<h2>Step 11: Train Tuned Random Forest Model</h2>
<pre>
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=3,
    random_state=42
)
model.fit(X_train, y_train)
</pre>

<h2>Step 12: Prediction</h2>
<pre>
y_pred = model.predict(X_test)
</pre>

<h2>Step 13: Evaluate Model</h2>
<pre>
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
</pre>

<h2>Model Performance</h2>
<ul>
  <li>Accuracy: ~86%</li>
  <li>Balanced dataset using SMOTE</li>
  <li>Random Forest model tuned for better performance</li>
</ul>

<h2>Golden Tip for You</h2>
<ul>
  <li>If you want, You can tune these parameters using <b> GridSearchCV </b> to see if you can push your Model's accuracy.
    This will find the best n_estimators, max_depth, etc. for your churn dataset.</li>
</ul>

<h2>Dataset</h2>
<p>Dataset used: <a href="https://www.kaggle.com/datasets/blastchar/telco-customer-churn?resource=download"><strong>Telco Customer Churn</strong></a> </p>
