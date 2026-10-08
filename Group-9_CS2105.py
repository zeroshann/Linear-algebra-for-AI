import numpy as np
import pandas as pd
from numpy.linalg import lstsq, inv
from numpy.linalg import solve, inv

A = np.array([
    [3,2,5,1],
    [4,1,4,2],
    [2,3,4,3],
    [5,2,3,1]
],dtype=float)

b = np.array([90,88,85,92],dtype=float)

weights = solve(A,b)

criteria=["GPA","Exam","Income","Leadership"]

print("Recovered Weights")
for c,w in zip(criteria,weights):
    print(f"{c:12s}: {w:.4f}")

print("\nVerification")
print(A@weights)
print("Original:",b)

new_student=np.array([1,2,3,4],dtype=float)
score=new_student@weights
print("\nAbacan, Maria Kathleen Joan:",new_student)
print("Predicted Scholarship Score:",round(score,2))

new_student=np.array([4,3,4,1],dtype=float)
score=new_student@weights
print("\nJobine Arago:",new_student)
print("Predicted Scholarship Score:",round(score,2))

new_student=np.array([4,3,2,1],dtype=float)
score=new_student@weights
print("\nJames Quito:",new_student)
print("Predicted Scholarship Score:",round(score,2))

new_student=np.array([4,4,3,1],dtype=float)
score=new_student@weights
print("\nJana Palacol:",new_student)
print("Predicted Scholarship Score:",round(score,2))

new_student=np.array([1,3,2,4],dtype=float)
score=new_student@weights
print("\nDenver Banayado:",new_student)
print("Predicted Scholarship Score:",round(score,2))

new_student=np.array([4,2,1,4],dtype=float)
score=new_student@weights
print("\nLance Gabaldon:",new_student)
print("Predicted Scholarship Score:",round(score,2))

new_student=np.array([4,2,3,2],dtype=float)
score=new_student@weights
print("\nKristian Carro :",new_student)
print("Predicted Scholarship Score:",round(score,2))

data = {
    "Student name":[ "Abacan, Maria Kathleen Joan", "Arago, Mary Jobine ", "Quito, James Andrey ", "Palacol, Jana Beatrice", "Banayado, Denver", "Gabaldon, Lance Gencedric", "Carro, John Kristian Shine" ],
    "GPA": [1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25],
    "Exam": [60, 52, 38, 44, 41, 12, 56],
    "Income": [100000, 70000, 20000, 50000, 20000, 70000, 30000],
    "Leadership": [10, 10, 7, 7, 3, 4, 8],
    "Scholarship": ["No", "Yes", "No", "Yes", "No", "No", "Yes"]
}

df = pd.DataFrame(data)

print("SCHOLARSHIP APPLICANT RECORDS")
print(df)