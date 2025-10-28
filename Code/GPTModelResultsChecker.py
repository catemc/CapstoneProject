import pandas as pd
import matplotlib.pyplot as plt
import io
import json

results_folder = "C:/Users/catem/OneDrive/Desktop/CapstoneProject/Results"

df = pd.read_csv('C:/Users/catem/OneDrive/Desktop/CapstoneProject/CombinedEMR+ManualCurations+PerColumn.csv')
subsetted_df = df[df['paper_id'].isin([
    "Kwon J. et al., 2018",
    ## "Nilsson B. et al., 2017", wrong info
    "Wang F. et al., 2015",
    "L'Huillier A. et al., 2015",
    "Lloren K. et al., 2019",
    "de Vries R. et al., 2017",
    "Sang X. et al., 2015b",
    "Baek Y. et al., 2015",
    "Xiao C. et al., 2016",
    ## "Suttie A. et al., 2019"
])]

# jsondf = pd.read_json('C:/Users/catem/OneDrive/Desktop/CapstoneProject/Results/e02467-16_results.json')
jsondf2 = pd.read_json('C:/Users/catem/OneDrive/Desktop/CapstoneProject/Results/embj0034-1661_results.json')
jsondf3 = pd.read_json('C:/Users/catem/OneDrive/Desktop/CapstoneProject/Results/jiv288_results.json')
jsondf4 = pd.read_json('C:/Users/catem/OneDrive/Desktop/CapstoneProject/Results/292_vir001029_results.json') 
jsondf5 = pd.read_json('C:/Users/catem/OneDrive/Desktop/CapstoneProject/Results/JVI.01825-18_results.json')
jsondf6 = pd.read_json('C:/Users/catem/OneDrive/Desktop/CapstoneProject/Results/ppat.1006390_results.json')
jsondf7 = pd.read_json('C:/Users/catem/OneDrive/Desktop/CapstoneProject/Results/srep15928_results.json')
jsondf8 = pd.read_json('C:/Users/catem/OneDrive/Desktop/CapstoneProject/Results/srep19474_results.json')
jsondf9 = pd.read_json('C:/Users/catem/OneDrive/Desktop/CapstoneProject/Results/zjv287_results.json')
# jsondf10 = pd.read_json('C:/Users/catem/OneDrive/Desktop/CapstoneProject/Results/11262_2019_Article_1700_review_results.json')

combined = pd.concat([jsondf2, jsondf3, jsondf4, jsondf5, jsondf6, jsondf7, jsondf8, jsondf9])
combined.to_csv('C:/Users/catem/OneDrive/Desktop/CapstoneProject/combinedoutput.csv', index=False)
combined["effect"] = combined["effect"].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
subsetted_df["mutation_name"] = subsetted_df["mutation_name"].str.replace(r"^[^:]+:", "", regex=True)

# Count how many times each mutation/effect pair occurs in the subset
subset_counts = subsetted_df.groupby(['mutation_name','effect_name']).size().to_dict()
subsetted_df.to_csv('C:/Users/catem/OneDrive/Desktop/CapstoneProject/subsettedoutput.csv', index=False)

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
print(subsetted_df[['mutation_name', 'effect_name', 'subtype']])

x = ['Matched', 'Unmatched', 'Undiscovered']
y = [count, no, excluded]
colors = ['green', 'yellow', 'red']
plt.bar(x,y, color=colors, zorder = 3)
plt.title("Comparison of FluMutDB and Extracted Mutations")
plt.ylabel('Counts')
plt.grid(True, linestyle='-', alpha=0.7, zorder = 0)
plt.show()
  
