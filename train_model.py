import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
import pickle

# Load the dataset
df = pd.read_csv("drug200.csv")

# Feature engineering: Binning Age and Na_to_K
bin_age = [0, 19, 29, 39, 49, 59, 69, 80]
category_age = ['<20s', '20s', '30s', '40s', '50s', '60s', '>60s']
df['Age_binned'] = pd.cut(df['Age'], bins=bin_age, labels=category_age)

bin_NatoK = [0, 9, 19, 29, 50]
category_NatoK = ['<10', '10-20', '20-30', '>30']
df['Na_to_K_binned'] = pd.cut(df['Na_to_K'], bins=bin_NatoK, labels=category_NatoK)

# Drop original columns
df = df.drop(['Age', 'Na_to_K'], axis=1)

# Define features and target
X = df.drop("Drug", axis=1)
y = df["Drug"]

# One-hot encode the categorical features
encoder = OneHotEncoder(sparse_output=False)
X_encoded = pd.DataFrame(encoder.fit_transform(X), columns=encoder.get_feature_names_out(X.columns))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=0)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=0)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Train the logistic regression model
model = LogisticRegression(max_iter=1000)  # optional: increase max_iter if needed
model.fit(X_train, y_train)

# Save the model and encoder
pickle.dump(model, open('finalModel.pkl', 'wb'))
pickle.dump(encoder, open('finalEncoder.pkl', 'wb'))

print("Model and encoder have been saved.")

