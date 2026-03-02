import pandas as pd


## Clean the food access data 
df = pd.read_excel("/Users/majiaoer/Desktop/final_project/data/raw/FoodAccessResearchAtlasData2019.xlsx", sheet_name="Food Access Research Atlas")
# Use these columns instead
df["CensusTract"] = df["CensusTract"].astype(str).str.split(".").str[0].str.zfill(11)

df_clean = df[[
    "CensusTract",
    "State",
    "County",
    "Urban",
    "LILATracts_1And10",     # food desert flag (1 mi urban / 10 mi rural)
    "LILATracts_halfAnd10",  # stricter definition
    "lapop1",                # pop beyond 1 mile from grocery (urban)
    "lapop10",               # pop beyond 10 miles from grocery (rural)
    "lalowi1",               # low income + low access, 1 mile
    "MedianFamilyIncome",
    "TractSNAP",             # SNAP households in tract
    "TractWhite",            # demographic controls
    "TractBlack",
    "TractHispanic",
    "TractHUNV",             # households with no vehicle
]]

df_clean.to_csv("data/derived/food_access_clean.csv", index=False)
print(f"Saved {len(df_clean)} tracts")


## Clean cdc places data
df_cdc = pd.read_csv("/Users/majiaoer/Desktop/final_project/data/raw/PLACES__Local_Data_for_Better_Health,_Census_Tract_Data,_2025_release_20260222.csv")
measures = ["DIABETES", "OBESITY", "BPHIGH", "CSMOKING"]
df_cdc = df_cdc[df_cdc["MeasureId"].isin(measures)]
df_cdc["LocationID"] = df_cdc["LocationID"].astype(str).str.zfill(11)

df_wide = df_cdc.pivot_table(index="LocationID", columns="MeasureId",
                          values="Data_Value", aggfunc="mean").reset_index()
df_wide.to_csv("data/derived/cdc_places_clean.csv", index=False)

