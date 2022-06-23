# -*- coding: utf-8 -*-
"""
@author: Malte Henningsen (Schomers)
"""
#%%
import RepSimAnalysis as ra
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
rearrange_order = [j*10+i for i in range(0,10) for j in range (0,3)]                       
in_folder = "" # insert appropriate path here
in_file = Path(in_folder, "raw_data_mean-firing-rates.txt")

RA = ra.RepSimAnalysis()
RA.DF = pd.read_csv(in_file, skiprows=0, sep='\\t', engine='python')
#%%
all_RDMs = RA.DF.groupby(['ModelNo','Type','AreaAbs',]).Neurons_FiringRates.\
                        apply(lambda x: x.values.tolist()).apply(\
                        lambda x: RA.get_rdm_cdist(x, rearrange=True))        

# 2nd-level: average across all 12 models
all_RDMs2 = all_RDMs.groupby(['Type','AreaAbs',]).\
            apply(lambda x: np.nanmean(np.rollaxis(np.dstack(x), -1), axis=0))
#%%
RA.saved_figs_default_folder = "path_to_saved_figs_folder"
RA.saved_figs_subfolder = ''     
RA.prefix_subtitles = ''
#%%
fname_list = []
title_dict = {'con50':'concrete LBL', 'con50sc': 'concrete NoLBL',
              'pair33':'abstract LBL', 'pair33sc': 'abstract NoLBL',}
RA.suptitle_y = 1.05
for name, value in list(all_RDMs2.groupby('Type')):
    fname = RA.show_12area_heatmap(value, title=title_dict.get(name),
                                   add_gdv=0, normalize=False,
                                   rank_transform=False, #True, #False
                                   scale=(0, 3.5), cmap=plt.get_cmap("hsv"),
                                   vmin=0., vmax=3.5,
                                   hpad=2.0, wpad=-10.0, figsize=(16,5.5),
                                   save_for_pub=True, central_areas_closer=True)
    fname_list.append(fname)    
#%%