'''
import lir central benchmark that jen created
get the 7mer sequence
remove the sequences that Jen added from the screen
drop duplicates
export to csv
'''
from lir_proteome_screen_pssm import environment as env
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path(env.DATA_DIR / "processed" / "test_sets")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


df = pd.read_csv(env.RAWFILEPATHS.lir_central_table)
df['7mer'] = df['Combined'].str[-7:]
df = df.drop_duplicates(subset=['7mer'])
df = df[
    [
        "7mer",
        "True label",
        "UNIPROT ACC_x",
        "UNIPROT ID_x",
        "Protein name",
        "Motif type",
        "Up-stream",
        "Motif",
        "Down-stream",
        "Start position",
        "End position",
        "Combined",
    ]
]
df = df.rename(
    columns = {
        "UNIPROT ACC_x": "UNIPROT_ACC",
        "UNIPROT ID_x": "UNIPROT_NAME",
        "Motif type": "_Motif type",
        "Up-stream": "_Up-stream",
        "Motif": "_Motif",
        "Down-stream": "_Down-stream",
        "Start position": "_Start position",
        "End position": "_End position",
        "Combined": "_Combined",
        "True label": "true label",
    }
)
df = df.dropna(subset=['UNIPROT_ACC'], axis=0)
df.to_csv(OUTPUT_DIR / "lir_central_test_set.csv", index=False)