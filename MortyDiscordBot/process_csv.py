import pandas as pd

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('RickAndMortyScripts.csv', sep='\t')

# Add a colon after every occurrence of "Rick" and "Morty" in the "line" column
df['line'] = df['line'].str.replace('Rick', 'Rick:').str.replace('Morty', 'Morty:')

# Write the modified DataFrame back to a CSV file
df.to_csv('output_file.csv', sep='\t', index=False)