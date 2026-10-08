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

new_student=np.array([1,3,5,4],dtype=float)
score=new_student@weights
print("\nJobine Arago:",new_student)
print("Predicted Scholarship Score:",round(score,2))

new_student=np.array([1,2,3,4],dtype=float)
score=new_student@weights
print("\nJana Beatrice Palacol:",new_student)
print("Predicted Scholarship Score:",round(score,2))

new_student=np.array([2,3,1,4],dtype=float)
score=new_student@weights
print("\nDenver Banayado:",new_student)
print("Predicted Scholarship Score:",round(score,2))

new_student=np.array([3,1,2,5],dtype=float)
score=new_student@weights
print("\nJames Andrey Quito:",new_student)
print("Predicted Scholarship Score:",round(score,2))

new_student=np.array([3,4,2,1],dtype=float)
score=new_student@weights
print("\nLance Gencedric Gabaldon:",new_student)
print("Predicted Scholarship Score:",round(score,2))

new_student=np.array([4,2,1,3],dtype=float)
score=new_student@weights
print("\nJohn Kristian Shine Carro:",new_student)
print("Predicted Scholarship Score:",round(score,2))