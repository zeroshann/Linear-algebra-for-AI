import numpy as np
import pandas as pd
from numpy.linalg import lstsq, inv
from numpy.linalg import solve, inv

np.set_printoptions(precision=4, suppress=True)

data = {
    "GPA": [1.25, 1.50, 2.75, 2.00, 1.30],
    "Exam": [95, 90, 70, 80, 96],
    "Income": [25000, 28000, 90000, 60000, 20000],
    "Leadership": [8, 9, 4, 6, 10],
    "Scholarship": ["Yes", "Yes", "No", "No", "Yes"]
}

df = pd.DataFrame(data)

print("SCHOLARSHIP APPLICANT RECORDS")
print(df)

A = np.array([
    [1.25, 95, 25000, 8],
    [1.50, 90, 28000, 9],
    [2.75, 70, 90000, 4],
    [2.00, 80, 60000, 6],
    [1.30, 96, 20000, 10]
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

print("\nCarro, John Kristian Shine")

new_student = np.array([1.25,56,30000,8])
prediction = new_student @ weights

print(Kristian)
print("\nSCHOLARSHIP PREDICTION")

print(f"Prediction Score: {prediction:.4f}")
if prediction >= 0.5:
    print("Scholarship Awarded: YES")
else:
    print("Scholarship Awarded: NO")
    
print("\nArago, Mary Jobine")

new_student = np.array([1.25,52,70000,9])
prediction = new_student @ weights

print(Jobine)
print("\nSCHOLARSHIP PREDICTION")

print(f"Prediction Score: {prediction:.4f}")
if prediction >= 0.5:
    print("Scholarship Awarded: YES")
else:
    print("Scholarship Awarded: NO")
    
print("\nPalacol, Jana Beatrice")

new_student = np.array([1.25,44,50000,9])
prediction = new_student @ weights

print(Jana)
print("\nSCHOLARSHIP PREDICTION")

print(f"Prediction Score: {prediction:.4f}")
if prediction >= 0.5:
    print("Scholarship Awarded: YES")
else:
    print("Scholarship Awarded: NO")
    
print("\nQuito, James Andrey")

new_student = np.array([1.25,38,20000,7])
prediction = new_student @ weights

print(Quito)
print("\nSCHOLARSHIP PREDICTION")

print(f"Prediction Score: {prediction:.4f}")
if prediction >= 0.5:
    print("Scholarship Awarded: YES")
else:
    print("Scholarship Awarded: NO")
    
print("\nBanayado, Denver")

new_student = np.array([1.25,41,20000,3])
prediction = new_student @ weights

print(Denver)
print("\nSCHOLARSHIP PREDICTION")

print(f"Prediction Score: {prediction:.4f}")
if prediction >= 0.5:
    print("Scholarship Awarded: YES")
else:
    print("Scholarship Awarded: NO")
    
print("\nGabaldon, Lance Gencedric")

new_student = np.array([1.25,12,60000,4])
prediction = new_student @ weights

print(Lance)
print("\nSCHOLARSHIP PREDICTION")

print(f"Prediction Score: {prediction:.4f}")
if prediction >= 0.5:
    print("Scholarship Awarded: YES")
else:
    print("Scholarship Awarded: NO")