import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Students_Grading_Dataset.csv")

# Handle missing values
df.dropna(inplace=True)

# Define the two groups
group1 = df[df['Internet_Access_at_Home'] == 'Yes']['Total_Score']
group2 = df[df['Internet_Access_at_Home'] == 'No']['Total_Score']

# Observed difference in means
observed_diff = np.mean(group1) - np.mean(group2)

# Number of permutations
num_permutations = 10_000

# Store permuted differences
perm_diffs = []

# Perform permutation test
combined = np.concatenate([group1, group2])
for _ in range(num_permutations):
    np.random.shuffle(combined)  # Shuffle the data
    new_group1 = combined[:len(group1)]
    new_group2 = combined[len(group1):]
    perm_diffs.append(np.mean(new_group1) - np.mean(new_group2))

# Compute p-value (two-tailed test)
p_value = np.sum(np.abs(perm_diffs) >= np.abs(observed_diff)) / num_permutations

# Plot the distribution of permuted differences
plt.hist(perm_diffs, bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.axvline(observed_diff, color='red', linestyle='dashed', linewidth=2, label='Observed Difference')
plt.axvline(-observed_diff, color='red', linestyle='dashed', linewidth=2)
plt.title("Permutation Test: Internet Access vs. Total Score")
plt.xlabel("Difference in Means")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# Print results
print(f"Observed Difference in Means: {observed_diff}")
print(f"P-value: {p_value}")

# Interpretation
if p_value < 0.05:
    print("Result: Reject the null hypothesis (significant difference found).")
else:
    print("Result: Fail to reject the null hypothesis (no significant difference).")

sns.set(style="whitegrid")

# 1. Scatter plot: Attendance vs. Total Score
plt.figure(figsize=(8,6))
sns.scatterplot(x='Attendance (%)', y='Total_Score', data=df, alpha=0.5)
plt.title('Attendance vs. Total Score')
plt.xlabel('Attendance (%)')
plt.ylabel('Total Score')
plt.show()

# 2. Box plot: Family Income Level vs. Total Score
plt.figure(figsize=(8,6))
sns.boxplot(x='Family_Income_Level', y='Total_Score', data=df)
plt.title('Family Income Level vs. Total Score')
plt.xlabel('Family Income Level')
plt.ylabel('Total Score')
plt.show()

# 3. Violin plot: Parent Education Level vs. Midterm Score
plt.figure(figsize=(8,6))
sns.violinplot(x='Parent_Education_Level', y='Midterm_Score', data=df)
plt.title('Parent Education Level vs. Midterm Score')
plt.xlabel('Parent Education Level')
plt.ylabel('Midterm Score')
plt.show()

# 4. Regression plot: Study Hours per Week vs. Total Score
plt.figure(figsize=(8,6))
sns.regplot(x='Study_Hours_per_Week', y='Total_Score', data=df, scatter_kws={"alpha":0.5})
plt.title('Study Hours per Week vs. Total Score')
plt.xlabel('Study Hours per Week')
plt.ylabel('Total Score')
plt.show()

# 5. Bar plot: Extracurricular Activities vs. Average Quiz Score
plt.figure(figsize=(8,6))
sns.barplot(x='Extracurricular_Activities', y='Quizzes_Avg', data=df)
plt.title('Extracurricular Activities vs. Average Quiz Score')
plt.xlabel('Extracurricular Activities (Yes/No)')
plt.ylabel('Average Quiz Score')
plt.show()