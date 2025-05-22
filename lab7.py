import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Read data from the CSV file
df = pd.read_csv("heart_data.csv")

model = BayesianNetwork(
    [
        ("age", "trestbps"),
        ("age", "fbs"),
        ("sex", "trestbps"),
        ("exang", "trestbps"),
        ("trestbps", "heartdisease"),
        ("fbs", "heartdisease"),
        ("heartdisease", "restecg"),
        ("heartdisease", "thalach"),
        ("heartdisease", "chol"),
    ]
)

model.fit(df, estimator=MaximumLikelihoodEstimator)
infer = VariableElimination(model)
q1 = infer.query(variables=["heartdisease"], evidence={"age": 63})
print(q1)

print("\n Inference: P(heartdisease | chol=233)")
result2 = infer.query(variables=["heartdisease"], evidence={"chol": 233})
print(result2)
