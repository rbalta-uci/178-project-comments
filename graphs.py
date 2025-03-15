import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data = pd.read_csv('train.csv')
toxic_types = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

plt.figure(figsize=(12, 6))

label_counts = data[toxic_types].sum().sort_values(ascending=False)
ax = sns.barplot(x=label_counts.index, y=label_counts.values)

for i, count in enumerate(label_counts.values):
    ax.text(i, count + 100, f"{count:,}", ha='center')

plt.title('Distribution of Toxicity')
plt.ylabel('Num of Comments')
plt.xlabel('Toxicity Types')
plt.savefig('label_distribution.png')
plt.close()
