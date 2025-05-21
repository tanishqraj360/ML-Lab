from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
model = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
predictions = model.predict(X_test)

for i, (pred, actual) in enumerate(zip(predictions, y_test)):
    print(
        f"{i + 1}. Predicted: {pred}, Actual: {actual}",
        "✅" if pred == actual else "❌",
    )
