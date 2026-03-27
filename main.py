import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# Step 1: Create Dataset

data = {
    "screen_time": [1,2,3,4,5,6,7,8,2,3,5,6,7,1,2,8,9,4,3,6],
    "sleep_hours": [8,7,6,7,5,4,3,2,8,7,5,4,3,9,8,2,2,6,7,4],
    "study_hours": [6,5,4,3,2,1,1,0,5,4,2,1,1,6,5,0,0,3,4,2],
    "productivity": [
        "High","High","Medium","Medium","Low","Low","Low","Low",
        "High","Medium","Low","Low","Low","High","High","Low","Low","Medium","Medium","Low"
    ]
}

df = pd.DataFrame(data)


# Step 2: Split Data

X = df[["screen_time", "sleep_hours", "study_hours"]]
y = df["productivity"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Step 3: Train Model

model = DecisionTreeClassifier()
model.fit(X_train, y_train)


# Step 4: Evaluate Model

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n📊 Model Accuracy:", round(accuracy * 100, 2), "%")


# Step 5: Feature Importance

importance = model.feature_importances_
features = ["Screen Time", "Sleep Hours", "Study Hours"]

print("\n🔍 Feature Importance:")
for f, imp in zip(features, importance):
    print(f"{f}: {round(imp*100,2)}%")


# Step 6: User Input System (FIXED - NO WARNING)

print("\n--- Screen Time Analyzer ---")

while True:
    try:
        screen = input("\nEnter screen time (hours) or 'exit': ")
        if screen.lower() == "exit":
            print("Exiting program...")
            break

        screen = float(screen)
        sleep = float(input("Enter sleep hours: "))
        study = float(input("Enter study hours: "))

        # FIXED INPUT (this removes warning completely)
        input_data = pd.DataFrame(
            [[screen, sleep, study]],
            columns=["screen_time", "sleep_hours", "study_hours"]
        )

        prediction = model.predict(input_data)[0]

        print("\n📊 Productivity Level:", prediction.upper())

        # Suggestions
        if prediction == "Low":
            print("⚠️ High screen time and low study detected.")
            print("👉 Reduce screen time and increase study/sleep.")
        elif prediction == "Medium":
            print("👍 Moderate balance.")
            print("👉 Try improving consistency in study habits.")
        else:
            print("🔥 Excellent routine!")
            print("👉 Keep maintaining this balance.")

    except:
        print("❌ Invalid input. Please enter numeric values.")