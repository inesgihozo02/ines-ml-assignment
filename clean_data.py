import pandas as pd

# ✅ Load Titanic dataset from URL
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

# ✅ Handle missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# ✅ Remove duplicates
df = df.drop_duplicates()

# ✅ Save cleaned data to file
df.to_csv("cleaned_data.csv", index=False)

print("Data cleaned and saved as cleaned_data.csv")


