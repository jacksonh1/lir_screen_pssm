# %%
import lir_proteome_screen_pssm.data_loaders as dl
import pandas as pd
import lir_proteome_screen_pssm.environment as env
import lir_proteome_screen_pssm.pssms as pssms
import matplotlib.pyplot as plt


# %%
background = dl.BGFREQS.proteome
foreground = dl.PROCESSED_SEQUENCE_TABLES.ilir_binders["7mer"].to_list()
for p in [0,0.1,1,10,100]:
    fg_counts = pssms.seqlist_2_counts_matrix(foreground, pseudocount=p)
    ilir_pssm = pssms.make_pssm(
        df_counts=fg_counts,
        bg=background
    )
    pssms.plot_logo(ilir_pssm)

fg_counts = pssms.seqlist_2_counts_matrix(foreground, pseudocount=0)
for aa in background.keys():
    fg_counts[aa] = fg_counts[aa] + background[aa]
ilir_pssm = pssms.make_pssm(
    df_counts=fg_counts,
    bg=background
)
pssms.plot_logo(ilir_pssm)
plt.show()
# %%
