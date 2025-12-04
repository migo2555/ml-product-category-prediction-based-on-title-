import joblib
import pandas as pd

model = joblib.load("model/Category_Label_model.pkl")

print("Model loaded successfully!")
print("Type 'exit' at any point to stop.\n")
 
while True:
    title = input(" Enter review title: ")
    if title.lower() == "exit":
        print("Exiting...")
        break

    title_word_count = len(title.split())
    digit_count = sum(c.isdigit() for c in title)

    user_input = pd.DataFrame([{
       "Product_Title": title,
       "brand": "unknown",
        "title_word_count": title_word_count,
        "digit_count": digit_count
    }]) 


    prediction = model.predict(user_input)[0]
    print(f"\nPredicted Category: {prediction}\n" + "-" * 40)