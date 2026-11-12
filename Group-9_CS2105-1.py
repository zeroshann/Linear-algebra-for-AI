import numpy as np
import pandas as pd
from numpy.linalg import lstsq, inv
from numpy.linalg import solve, inv

np.set_printoptions(precision=4, suppress=True)

data = {
    "Name": ["Carro, John Kristian Shine", "Arago, Mary Jobine","Palacol, Jana Beatrice","Quito, James Andrey", "Banayado, Denver", "Gabaldon, Lance Gencedric"],
    "GPA": [1.25, 1.25, 1.25, 1.25, 1.25, 1.25],
    "Exam": [56, 52, 44, 38, 41, 12],
    "Income": [30000, 70000, 50000, 20000, 20000, 60000],
    "Leadership": [8, 9, 7, 9, 3, 4],
    "Scholarship": ["Yes", "Yes", "No", "No", "Yes", "No"]
}

df = pd.DataFrame(data)

print("SCHOLARSHIP APPLICANT RECORDS")
print(df)

A = np.array([
    [1.25, 56, 30000, 8],
    [1.25, 52, 70000, 9],
    [1.25, 44, 50000, 7],
    [1.25, 38, 20000, 9],
    [1.25, 41, 20000, 3],
    [1.25, 12, 60000, 4]
], dtype=float)

print("A = Matrix A")
print("Matrix A:")
print(A)

Scholarship_binary=[1.0 if val== "Yes" else 0.0 for val in data["Scholarship"]]

B = np.array(Scholarship_binary, dtype=float).reshape(-1,1)
b = np.array(Scholarship_binary, dtype=float)

print("B = Matrix B")
print("Matrix B:")
print(B)

weights, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
weights = weights.flatten()

criteria = ["GPA", "Exam", "Income", "Leadership"]

print("\nMODEL WEIGHTS")
for criterion, weight in zip(criteria, weights):
    print(f"{criterion:12s}: {weight:.6f}")

print("\nVerification")
predictions = A @ weights
print("Raw scores:", predictions)
print("Original:",b)

Kristian = pd.DataFrame ({
    "GPA": [1.25],
    "Exam": [56],
    "Income": [30000],
    "Leadership": [8]
})

Jobine = pd.DataFrame ({
    "GPA": [1.25],
    "Exam": [52],
    "Income": [70000],
    "Leadership": [9]
})

Jana = pd.DataFrame ({
    "GPA": [1.25],
    "Exam": [44],
    "Income": [50000],
    "Leadership": [9]
})

Quito = pd.DataFrame ({
    "GPA": [1.25],
    "Exam": [38],
    "Income": [20000],
    "Leadership": [7]
})

Denver = pd.DataFrame ({
    "GPA": [1.25],
    "Exam": [41],
    "Income": [20000],
    "Leadership": [3]
})

Lance = pd.DataFrame ({
    "GPA": [1.25],
    "Exam": [12],
    "Income": [60000],
    "Leadership": [4]
})

print("\n Name: Carro, John Kristian Shine")

new_student = np.array([1.25,56,30000,8])
prediction = new_student @ weights

print(Kristian)
print("\nSCHOLARSHIP PREDICTION")

print(f"Prediction Score: {prediction:.4f}")
if prediction >= 0.5:
    print("Scholarship Awarded: YES")
else:
    print("Scholarship Awarded: NO")
    
print("\n Name: Arago, Mary Jobine")

new_student = np.array([1.25,52,70000,9])
prediction = new_student @ weights

print(Jobine)
print("\nSCHOLARSHIP PREDICTION")

print(f"Prediction Score: {prediction:.4f}")
if prediction >= 0.5:
    print("Scholarship Awarded: YES")
else:
    print("Scholarship Awarded: NO")
    
print("\n Name: Palacol, Jana Beatrice")

new_student = np.array([1.25,44,50000,9])
prediction = new_student @ weights

print(Jana)
print("\nSCHOLARSHIP PREDICTION")

print(f"Prediction Score: {prediction:.4f}")
if prediction >= 0.5:
    print("Scholarship Awarded: YES")
else:
    print("Scholarship Awarded: NO")
    
print("\n Name: Quito, James Andrey")

new_student = np.array([1.25,38,20000,7])
prediction = new_student @ weights

print(Quito)
print("\nSCHOLARSHIP PREDICTION")

print(f"Prediction Score: {prediction:.4f}")
if prediction >= 0.5:
    print("Scholarship Awarded: YES")
else:
    print("Scholarship Awarded: NO")
    
print("\n Name: Banayado, Denver")

new_student = np.array([1.25,41,20000,3])
prediction = new_student @ weights

print(Denver)
print("\nSCHOLARSHIP PREDICTION")

print(f"Prediction Score: {prediction:.4f}")
if prediction >= 0.5:
    print("Scholarship Awarded: YES")
else:
    print("Scholarship Awarded: NO")
    
print("\n Name: Gabaldon, Lance Gencedric")

new_student = np.array([1.25,12,60000,4])
prediction = new_student @ weights

print(Lance)
print("\nSCHOLARSHIP PREDICTION")

print(f"Prediction Score: {prediction:.4f}")
if prediction >= 0.5:
    print("Scholarship Awarded: YES")
else:
    print("Scholarship Awarded: NO")