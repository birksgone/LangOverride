import pandas as pd
try:
    df_check = pd.read_csv("hero_skill_output.csv")
    print("CSV file is valid and readable by pandas.")
except Exception as e:
    print(f"Error reading CSV: {e}")