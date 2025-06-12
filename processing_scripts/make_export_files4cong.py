# %%
import lir_proteome_screen_pssm.data_loaders as dl
import pandas as pd
import lir_proteome_screen_pssm.environment as env
import lir_proteome_screen_pssm.pssms as pssms
import matplotlib.pyplot as plt

EXPORT_FOLDER = env.DATA_DIR / 'exported_tables'
EXPORT_FOLDER.mkdir(parents=True, exist_ok=True)

# %%

# ==============================================================================
# // tables
# ==============================================================================


# lir_central = dl.TEST_SETS.lir_central.copy()
# lir_central[['7mer', 'true label', 'UNIPROT_ACC', '_Combined']].to_csv(EXPORT_FOLDER / 'lir_central.csv', index=False)

lir_central_augmented = dl.TEST_SETS.lir_central_augmented.copy()
lir_central_augmented[['7mer', 'true label', 'UNIPROT_ACC', '_Combined']].to_csv(EXPORT_FOLDER / 'lir_central_augmented.csv', index=False)

screen_binders = dl.PROCESSED_SEQUENCE_TABLES.screen_binders.copy()
screen_binders[["7mer", "true label", "avg_z_score", "lir_type", "_ID"]].to_csv(EXPORT_FOLDER / 'screen_binders.csv', index=False)

screen_nonbinders = dl.PROCESSED_SEQUENCE_TABLES.screen_nonbinders.copy()
screen_nonbinders[["7mer", "true label", "avg_z_score", "lir_type", "_ID"]].to_csv(EXPORT_FOLDER / 'screen_nonbinders.csv', index=False)


ilir = dl.PROCESSED_SEQUENCE_TABLES.ilir_binders.copy()
ilir["true label"] = 1
ilir[["7mer", "true label"]].to_csv(EXPORT_FOLDER / 'ilir_binders.csv', index=False)


# ==============================================================================
# // ilir PSSMs
# ==============================================================================
# %%

background = dl.BGFREQS.proteome
foreground = dl.PROCESSED_SEQUENCE_TABLES.ilir_binders["7mer"].to_list()

for pcount in [0, 1]:
    fg_counts = pssms.seqlist_2_counts_matrix(foreground, pseudocount=pcount)
    ilir_pssm = pssms.make_pssm(
        df_counts=fg_counts,
        bg=background
    )
    ilir_pssm.to_csv(EXPORT_FOLDER / f'ilir_pssm-pseudocount_{pcount}.csv')
    fig, ax = plt.subplots(figsize=(10, 4))
    pssms.plot_logo(ilir_pssm, ax = ax, title=f'ilir PSSM; pseudocount = {pcount}')
    plt.savefig(EXPORT_FOLDER / f'ilir_pssm-pseudocount_{pcount}.png', dpi=300, bbox_inches='tight')


