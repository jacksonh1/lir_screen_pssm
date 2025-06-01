from lir_proteome_screen_pssm import environment as env
from lir_proteome_screen_pssm import pssms
import pandas as pd
import numpy as np
import logomaker as lm
import matplotlib.pyplot as plt
plt.style.use('custom_standard')
plt.style.use('custom_small')
import seaborn as sns
import lir_proteome_screen_pssm.sequence_utils as seqtools

from pathlib import Path


# ==============================================================================
# // 2025 proteome
# ==============================================================================
proteome_path = env.DATA_DIR / 'raw' / 'human_proteome_DB' / 'UP000005640_9606.fasta'
faimporter = seqtools.FastaImporter(proteome_path)
proteome = faimporter.import_as_list()
proteome_str = ''.join([str(s.seq) for s in proteome])
proteome_bg = seqtools.get_residue_frequencies(proteome_str)
bg_freqs = pd.DataFrame.from_dict(proteome_bg, orient='index', columns=['proteome']).reset_index(names='Residue')



# ==============================================================================
# // input library
# ==============================================================================
df = pd.read_csv(env.RAWFILEPATHS.full_screening_table, sep='\t')
df = df[df['Input Count'] > 100]
screen_input_bg_str = ''.join(df['ID'].unique())
screen_input_bg = seqtools.get_residue_frequencies(screen_input_bg_str)
bg_freqs['input_library'] = bg_freqs['Residue'].map(screen_input_bg)


# ==============================================================================
# // ilir reported background frequencies
# ==============================================================================
# from a previous notebook
ilir_bg = {
    'M': 0.02132438371877552,
    'L': 0.09963465604719016,
    'P': 0.06315319257172128,
    'K': 0.05732637839353644,
    'N': 0.03589313658554801,
    'R': 0.05636127568998062,
    'I': 0.04337783480203681,
    'A': 0.07013252855851457,
    'H': 0.026225904138650682,
    'E': 0.07103976737484315,
    'F': 0.03649449826285036,
    'G': 0.06574165474159767,
    'V': 0.05963890820620893,
    'D': 0.04737264484290967,
    'Q': 0.04769437510172419,
    'S': 0.08328850423348756,
    'C': 0.023007985040291584,
    'W': 0.012147453037774814,
    'Y': 0.026631039422129122,
    'T': 0.05351070860627121,
    'U': 3.170623957657374e-06
}
bg_freqs['ilir_bg'] = bg_freqs['Residue'].map(ilir_bg)


# ==============================================================================
# // non-binder background frequencies
# ==============================================================================
sc_df = pd.read_csv(env.RAWFILEPATHS.screening_hits_table)
nonbinders = sc_df[sc_df['Bind/Nonbind'] == 'Nonbind'].copy()
nonbinders = nonbinders['first_8_residues'].to_list()
c_nbs = pssms.alignment_2_counts(nonbinders) # residue counts
# take all the undefined positions from the non-binder set (all except 4 and 7)
undefined_positions = c_nbs.loc[[i for i in c_nbs.index if i != 4 and i!=7], :].copy()
# sum the counts for every residue and divide by the total number of counts
nonbinder_bg_frequency = (undefined_positions.sum()/undefined_positions.sum().sum())
bg_freqs['nonbinders_undef_pos'] = bg_freqs['Residue'].map(nonbinder_bg_frequency)


# ==============================================================================
# // export
# ==============================================================================
bg_frequency_path = env.DATA_DIR / 'processed' / 'background_frequencies.csv'
bg_freqs.to_csv(bg_frequency_path, index=False)

