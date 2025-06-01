from lir_proteome_screen_pssm import environment as env
import pandas as pd

output_dir = env.PROCESSED_DATA_DIR
output_dir.mkdir(exist_ok=True, parents=True)


rename_dict = {
    "first_4_residues": "4mer",
    "first_6_residues": "6mer",
    "first_8_residues": "8mer",
    "first_14_residues": "14mer",
    "first_7_residues": "7mer",
    "first_5_residues": "5mer",
}

screen_df = pd.read_csv(env.RAWFILEPATHS.screening_hits_table)
screen_df = screen_df.rename(columns=rename_dict)
assert all(
    [len(s) == 7 for s in screen_df["7mer"].tolist()]
), "All binders should be 7 residues long"
assert all(
    screen_df["7mer"].str.contains('-') == False
), "No 7mers should contain gaps"

hide_col_dict = {}
for col in screen_df.columns:
    if col in list(rename_dict.values()) + ["avg_z_score"]:
        continue
    hide_col_dict[col] = "_" + col
screen_df = screen_df.rename(columns=hide_col_dict)

binders_df = screen_df[screen_df["_Bind/Nonbind"] == "Bind"].copy()
nonbinders = screen_df[screen_df['_Bind/Nonbind'] == 'Nonbind'].copy()
binders_df.to_csv(output_dir / "screen-binders.csv", index=False)
nonbinders.to_csv(output_dir / "screen-nonbinders.csv", index=False)
