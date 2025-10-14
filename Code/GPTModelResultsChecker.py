
import pandas as pd
import matplotlib.pyplot as plt
import io
import json

results_folder = "C:/Users/catem/OneDrive/Desktop/CapstoneProject/Results"

df = pd.read_csv('C:/Users/catem/OneDrive/Desktop/CapstoneProject/CombinedEMR+ManualCurations+PerColumn.csv')
subsetted_df = df[df['paper_id'].isin([
    "Lan Y. et al., 2010",
    "Ilyushina N. et al., 2005",
    "Choi W. et al., 2013",
    "Xiao C. et al., 2016",
    "Baek Y. et al., 2015",
    "Wang W. et al., 2010",
    "Hurt A. et al., 2007",
    "Chutinimitkul S. et al., 2010",
    "Suttie A. et al., 2019",
    "Cheung C. et al., 2006"
])]

jsondf = pd.read_json('C:/Users/catem/OneDrive/Desktop/CapstoneProject/Results/0221-10_results.json')
jsondf2 = pd.read_json('C:/Users/catem/OneDrive/Desktop/CapstoneProject/Results/0334-09_results.json')
jsondf3 = pd.read_json('C:/Users/catem/OneDrive/Desktop/CapstoneProject/Results/2737-09_results.json')
jsondf4 = pd.read_json('C:/Users/catem/OneDrive/Desktop/CapstoneProject/Results/11262_2019_Article_1700_review_results.json') 
jsondf5 = pd.read_json('C:/Users/catem/OneDrive/Desktop/CapstoneProject/Results/37883998_results.json')
jsondf6 = pd.read_json('C:/Users/catem/OneDrive/Desktop/CapstoneProject/Results/lan-et-al-2010-a-comprehensive-surveillance-of-adamantane-resistance-among-human-influenza-a-virus-isolated-from_results.json')
jsondf7 = pd.read_json('C:/Users/catem/OneDrive/Desktop/CapstoneProject/Results/main (1)_results.json')
jsondf8 = pd.read_json('C:/Users/catem/OneDrive/Desktop/CapstoneProject/Results/main_results.json')
jsondf9 = pd.read_json('C:/Users/catem/OneDrive/Desktop/CapstoneProject/Results/srep19474_results.json')
jsondf10 = pd.read_json('C:/Users/catem/OneDrive/Desktop/CapstoneProject/Results/zjv287_results.json')

combined = pd.concat([jsondf, jsondf2, jsondf3, jsondf4, jsondf5, jsondf6, jsondf7, jsondf8, jsondf9, jsondf10])
combined.to_csv('C:/Users/catem/OneDrive/Desktop/CapstoneProject/combinedoutput.csv', index=False)
combined["effect"] = combined["effect"].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
subsetted_df["mutation_name"] = subsetted_df["mutation_name"].str.replace(r"^[^:]+:", "", regex=True)

count = 0
no = 0
for i in range(len(combined)):
    short_mutation = combined['mutation'].iloc[i]
    effect_grab = combined['effect'].iloc[i]
    if isinstance(effect_grab, list):
        effect_grab = ", ".join(effect_grab)
    exists = ((subsetted_df["mutation_name"] == short_mutation) & (subsetted_df['effect_name'] == effect_grab)).any()
    if exists:
        count += 1
    else:
        no += 1

print(count)
print(no)
excluded = len(subsetted_df) - (count + no)
print(len(subsetted_df) - (count + no))

x = ['Matched', 'Unmatched', 'Undiscovered']
y = [count, no, excluded]
colors = ['green', 'yellow', 'red']
plt.bar(x,y, color=colors, zorder = 3)
plt.title("Comparison of FluMutDB and Extracted Mutations")
plt.ylabel('Counts')
plt.grid(True, linestyle='-', alpha=0.7, zorder = 0)
plt.show()
  