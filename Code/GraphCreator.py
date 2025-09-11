import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('C:/Users/catem/OneDrive/Desktop/CapstoneProject/CombinedEffectsandMutations.csv')
sorted_counts = df['effect_name'].value_counts().sort_values(ascending=False)#.head(50)
df_sorted = sorted_counts.sort_values(ascending=True)

df_sorted.plot.barh(figsize=(6, 20))
plt.tick_params(axis='y', labelsize=4) # change to 7 if you want to do top 50
plt.xlabel('Counts')
plt.ylabel('Type of Effect')
plt.title('Effects of Mutations on Strains')

plt.subplots_adjust(left=0.3)  # increase left margin
plt.show()