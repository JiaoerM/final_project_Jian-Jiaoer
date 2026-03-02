import pandas as pd

food = pd.read_csv("/Users/majiaoer/Desktop/final_project/data/derived/food_access_clean.csv", dtype={"CensusTract": str})
cdc  = pd.read_csv("/Users/majiaoer/Desktop/final_project/data/derived/cdc_places_clean.csv", dtype={"LocationID": str})

merged = food.merge(cdc, left_on="CensusTract", right_on="LocationID", how="inner")
merged.to_csv("/Users/majiaoer/Desktop/final_project/data/derived/merged_tract_data.csv", index=False)
print(f"Merged: {len(merged)} tracts")