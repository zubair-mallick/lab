import numpy as np
import pandas as pd

# Load data
data = pd.read_csv('training_data.csv')
print("\nDataset:\n", data)

# Split features and target
concepts = np.array(data.iloc[:, :-1])    # all columns except last
target = np.array(data.iloc[:, -1])       # only last column

print("\nConcepts:\n", concepts)
print("\nTarget:\n", target)

# Candidate Elimination Algorithm
def candidate_elimination(concepts, target):
    specific_h = concepts[0].copy()
    general_h = [["?" for _ in specific_h] for _ in specific_h]

    print("\nInitial Specific_h:", specific_h)
    print("Initial General_h:", general_h)

    for i, row in enumerate(concepts):
        if target[i].lower() == "yes":
            for j in range(len(specific_h)):
                if row[j] != specific_h[j]:
                    specific_h[j] = "?"
                    general_h[j][j] = "?"
        else:  # target is "no"
            for j in range(len(specific_h)):
                if row[j] != specific_h[j]:
                    general_h[j][j] = specific_h[j]
                else:
                    general_h[j][j] = "?"

        print(f"\nStep {i+1}")
        print("Specific_h:", specific_h)
        print("General_h:", general_h)

    # Remove fully generic hypotheses
    general_h = [g for g in general_h if g != ["?" for _ in specific_h]]
    return specific_h, general_h

# Run
s_final, g_final = candidate_elimination(concepts, target)

# Results
print("\nFinal Specific_h:", s_final)
print("\nFinal General_h:", g_final)
