import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import joblib
 
# Load dataset directly from GitHub
url = "https://raw.githubusercontent.com/migo2555/ml-product-category-prediction-based-on-title-/main/data/TASK_03-products.csv"
df = pd.read_csv(url)

df.columns = df.columns.str.strip()
 
# rename columns for easier access
df = df.rename(columns={
    "Product Title": "Product_Title",
    "Category Label": "Category_Label",
    "_Product Code": "_Product_Code",
    "Listing Date": "Listing_Date"
})

# Drop all rows with missing values
df = df.dropna(subset=['Category_Label', 'Product_Title'])

#  Standardize target and text columns
df["Category_Label"] = df["Category_Label"].str.lower().str.strip()

df["Product_Title"] = df["Product_Title"].str.lower().str.strip()

 # Extract brand as first word of product title
df['brand'] = df['Product_Title'].apply(lambda x: x.split()[0] if isinstance(x, str) else "unknow")

# Count number of words in product title
df['title_word_count'] = df['Product_Title'].apply(lambda x: len(str(x).split())) 

# Count number of digits in product title
df['digit_count'] = df['Product_Title'].apply(lambda x: sum(c.isdigit() for c in str(x)))

# Create new column with length of each review_text
df['review_length'] = df['Product_Title'].astype(str).str.len()
 
# Define features and label
X = df[[
    "Product_Title",
    "brand",
    "title_word_count",
    "digit_count"
]]
y = df["Category_Label"]

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("title", TfidfVectorizer(), "Product_Title"),
        ("brand", OneHotEncoder(handle_unknown="ignore"), ["brand"]),
        ("numeric", MinMaxScaler(), ["title_word_count", "digit_count"])
    ]
)

# Define pipeline with the best model (Support Vector Machine)
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LinearSVC())
])

# Train the model on the entire dataset
pipeline.fit(X, y)

# Save the model to a file
joblib.dump(pipeline, "model/Category_Label_model.pkl")

print("Model trained and saved as 'model/Category_Label_model.pkl'")

