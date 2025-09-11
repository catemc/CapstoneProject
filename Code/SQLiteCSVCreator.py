import sqlite3
import pandas as pd

conn = sqlite3.connect('C:/Users/catem/OneDrive/Desktop/CapstoneProject/flumut_db.sqlite')

mutations = pd.read_sql_query("SELECT * FROM markers_mutations", conn) #originally markers_mutations
effects = pd.read_sql_query("SELECT * FROM markers_effects", conn) #markers_effects seems most promising
effects_dropped = effects.drop(['in_vivo', 'in_vitro'], axis=1)
merged_df = pd.merge(mutations, effects_dropped, on='marker_id')

merged_df.to_csv('C:/Users/catem/OneDrive/Desktop/CapstoneProject/CombinedEffectsandMutations.csv', index=False)
mutations.to_csv('C:/Users/catem/OneDrive/Desktop/CapstoneProject/MarkerMutations.csv', index=False)
effects_dropped.to_csv('C:/Users/catem/OneDrive/Desktop/CapstoneProject/MarkerEffects.csv', index=False)


conn.close()
