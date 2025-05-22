import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

data_dict = {
    "age": [63, 67, 67, 41, 62, 60],
    "sex": [1, 1, 1, 0, 0, 1],
    "trestbps": [145, 160, 120, 130, 140, 130],
    "chol": [233, 286, 229, 204, 268, 206],
    "fbs": [1, 0, 0, 0, 0, 0],
    "restecg": [2, 2, 2, 2, 2, 2],
    "thalach": [150, 108, 129, 172, 160, 132],
    "exang": [0, 1, 1, 0, 0, 1],
    "heartdisease": [0, 2, 1, 0, 3, 4],
}
df = pd.DataFrame(data_dict)

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
