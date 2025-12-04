# ml-product-category-prediction-based-on-title

# ml-product-category-prediction-based-on-title

This repository contains an end-to-end machine learning project for predicting product categories based on product titles. 
The project includes data exploration, cleaning, feature engineering, model training, and evaluation.

The repository structure is as follows:

- `data/` — contains `TASK_03-products.csv`
- `notebooks/` — contains `product_category_prediction.ipynb`
- `src/` — contains `train_model.py` and `test_model.py`
- `.gitignore` — specifies files to ignore

## 1. Dataset

**Location:** `/data/TASK_03-products.csv`

The dataset contains product listings with the following attributes:

- **Product Title** – the name or title of the product  
- **Number of Views** – how many times the product has been viewed  
- **Merchant Rating** – the rating of the seller  
- **Listing Date** – date when the product was listed  
- **Product Category (label)** – the target variable for prediction  
- **Additional metadata** – other relevant product information  

This dataset is used to train a machine learning model for **automatic product category classification**.

## 2. Jupyter Notebook

**File:** `notebooks/product_category_prediction.ipynb`

The notebook was developed in **Google Colab** and contains the complete machine learning workflow for predicting product categories.  

It includes the following steps:

1. **Data Loading & Exploration** – loading dataset, inspecting structure, checking missing values.  
2. **Data Cleaning & Preprocessing** – standardizing column names, handling missing values, cleaning text fields, converting date columns.  
3. **Feature Engineering** – extracting textual features, binary flags for keywords, brand extraction, character & word counts.  
4. **Exploratory Visualizations** – outlier detection, category distribution, cleaning inconsistent labels, brand-category frequency analysis.  
5. **Model Training & Evaluation** – testing Logistic Regression, Decision Tree, Random Forest, and SVM; selecting SVM as the best-performing model.  
6. **Final Conclusion** – dataset prepared and cleaned; SVM chosen for further optimization and deployment.

## 3. Source Code (`src` folder)

### `train_model.py`
Contains the full training pipeline, including:

- **Data preprocessing** – handling missing values, cleaning, and transforming features  
- **Feature engineering** – creating additional features from product titles and metadata  
- **Encoding and scaling** – encoding categorical variables and scaling numerical features  
- **Model training** – training the Support Vector Machine (SVM) model by default  
- **Saving the trained model** – storing the trained model for later use  

### `test_model.py`
This script is used to:

- **Load the saved model**  
- **Run predictions** on new examples  
- **Evaluate model performance** using appropriate metrics  

These scripts are useful for model deployment or further development/testing.

## 4. Old Notebook (optional for deletion)

If your repository contains an additional notebook with only 7 steps (e.g., an earlier version from Colab), it is **safe to delete**, since it is incomplete and has already been replaced by the full notebook:  

`notebooks/product_category_prediction.ipynb`

## 5. Final Conclusion

After loading, inspecting, and cleaning the dataset, the data is now fully prepared for modeling. Missing values were handled, column names standardized, inconsistencies removed, and additional text-based and numerical features engineered. The exploratory analysis of categories, brands, and outliers provided a clear understanding of the dataset’s structure and potential challenges.

Following the training of multiple models (Logistic Regression, Decision Tree, Random Forest, and SVM), **Support Vector Machine (SVM)** achieved the best overall performance.  

SVM will therefore be used as the **primary model** in the next steps for further optimization and feature refinement.

## 6. How to Run the Project

### 1. Install dependencies
```bash
pip install -r requirements.txt

python src/train_model.py

python src/test_model.py

### Notes

- If `old_notebook.ipynb` contains only 7 steps, it should be **removed** to keep the repository clean.  
- All important implementation work is contained in the full notebook (`product_category_prediction.ipynb`) and the `src` scripts.
