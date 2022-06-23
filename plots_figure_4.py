# -*- coding: utf-8 -*-
"""
@author: Malte Henningsen (Schomers)
"""
import RepSimAnalysis as ra
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import SeabornFig2Grid as sfg
from pathlib import Path

##############################################################################################
## some general variables ####################################################################
area_names_dict = dict(zip(range(1,13), ['V1','TO','AT','PF$_L$','PM$_L$','M1$_L$',                     
                                         'A1','AB','PB','PF$_i$','PM$_i$','M1$_i$',]))
area_colors_dict = ['', '#117531','#4CC45A','#AEEDAB', '#EEE04E', '#FB9C3F', '#9B5615',
                        '#3A439F','#409FC0','#A4E0EC', '#E04197', '#FF9792', '#E43A2F']
# Note that GDV_rel and GDV_ur is old terminology
# GDV_rel (related) is called Dissim-within in the paper
# GDV_ur (unrelated) is called Dissim-between in the paper
depvals = ['GDV_rel', 'GDV_ur']     # dependent variables
##############################################################################################
#%%
def multipleReplace(text, wordDict):
    for key in wordDict:
        text = text.replace(key, wordDict[key])
    return text
#%%
plot_nolabels_only = False
shared_name = 'GDV_type'
type_name_rplc_dict = {'pair33': 'abstract', 'con50': 'concrete',}
_id_factors_dict = {'TM': ['Type','Model_ID',], 'MT': ['Model_ID','Type',],}
id_factors = _id_factors_dict['TM']
hue_vs_x = {'x': 'Type_2', 'hue': 'Type_1'}
hue_vs_x_palette = ['#ff7878','#7878ff',]
hue_order_list = {'Type': [multipleReplace(s, type_name_rplc_dict) for s in ['pair33','con50',]],
                  'GDV_type': [multipleReplace(s, type_name_rplc_dict) for s in ['pair33',
                               'pair33sc','con50','con50sc',]],
                  }.get('GDV_type') 
                  
#%%
RA = ra.RepSimAnalysis()
in_folder = "" # insert appropriate path here
in_file = Path(in_folder, "raw_data_mean-firing-rates.txt")
RA.DF = pd.read_csv(in_file, skiprows=0, sep='\\t', engine='python')
all_RDMs = RA.DF.groupby(['Model_ID','Type','AreaAbs',]).Neurons_FiringRates.\
                        apply(lambda x: x.values.tolist()).apply(\
                        lambda x: RA.get_rdm_cdist(x, rearrange=True)) 
                        
#%%
# calculate DissimB and DissimW
dissim_vals = all_RDMs.groupby(['Type','AreaAbs','Model_ID',]).apply(lambda x: ra.dv_from_dm(x[0], rearrange=True,
                                           return_as_tuple=True, normalize=False))
res = pd.DataFrame(dissim_vals.values.tolist(), index=dissim_vals.index)
res.rename(columns={0:'GDV_rel', 1: 'GDV_ur'}, inplace=True)
res = res.reset_index()

#%%
##############################
# Figure 4A ##################
##############################
# parameters for 6x2 plot:
left   =  0.1    # the left side of the subplots of the figure
right  =  0.9    # the right side of the subplots of the figure
bottom =  0.05   # the bottom of the subplots of the figure
top    =  0.95   # the top of the subplots of the figure
wspace =  .3     # the amount of width reserved for blank space between subplots
hspace =  .9     # the amount of height reserved for white space between subplots
# The amount of space above titles
y_title_margin = .9
    
# This function adjusts the sub plots using the above paramters
plt.subplots_adjust(
    left    =  left, 
    bottom  =  bottom, 
    right   =  right, 
    top     =  top, 
    wspace  =  wspace, 
    hspace  =  hspace
)

font_sizes = [19,21,23,25]
y_axis_limits = (0,2.5)
g=[None]*12
for i in range(1,13):        
    y_var_name = "GDV"    
    dplt = res[res.AreaAbs==i].groupby(id_factors).mean().\
        loc[:, np.r_[depvals]].reset_index()      
    dplt['Diff_RU'] = dplt['GDV_ur']-dplt['GDV_rel']      
        
    dplt['Type_1'] = np.nan
    dplt.loc[dplt.Type.str.startswith('pair33'), 'Type_1'] = 'pair33'
    dplt.loc[dplt.Type.str.startswith('con50'), 'Type_1'] = 'con50'
    dplt['Type_2'] = 'Label' 
    dplt.loc[dplt.Type.str.endswith('sc'), 'Type_2'] = 'NoLabel'
    
    dplt['Type'] = dplt['Type'].replace(type_name_rplc_dict, regex=True)
    dplt['Type_1'] = dplt['Type_1'].replace(type_name_rplc_dict, regex=True)
    tmp_legend = [1,0] if i==6 else [0,0]  # add legend to top right plot
    hue_order_list = ['GDV_rel', 'GDV_ur']
    this_hue = hue_vs_x['hue']
    x_order_list = ['Label','NoLabel',]
    y_var_name='Diff_RU'   # =DissimDiff
    hue_order_list=['abstract','concrete']
    if plot_nolabels_only:
        dplt = dplt[dplt['Type_2']!='Label']
        g[i-1] = sns.catplot( data=dplt, x='Type_1', y=y_var_name,
            palette=['b','r'], ci=95,
            legend=tmp_legend[0], legend_out=tmp_legend[1], kind='bar')
    else:
        g[i-1] = sns.catplot( data=dplt, x=hue_vs_x['x'], y=y_var_name,
            palette=hue_vs_x_palette, order=x_order_list,
            legend=tmp_legend[0], legend_out=tmp_legend[1], kind='bar',
            hue=this_hue, ci=95,
            hue_order=hue_order_list )
    if i==6:
        plt.legend(title='Semantic Type',
                   title_fontsize=font_sizes[0],
                   fontsize=font_sizes[0],
                   bbox_to_anchor=(1.32, 0.91))
    g[i-1].fig.get_axes()[0].set_ylim(y_axis_limits)
    g[i-1].fig.get_axes()[0].set_title(area_names_dict[i], fontsize=font_sizes[0])
    g[i-1].fig.get_axes()[0].set_title("         " + area_names_dict[i] + "         ",
                                         y=.94, fontsize=font_sizes[1])
    g[i-1].fig.get_axes()[0].title.set_backgroundcolor(area_colors_dict[i])
    g[i-1].fig.get_axes()[0].set_xlabel('')
    g[i-1].fig.get_axes()[0].tick_params(axis='y', which='major', labelsize=font_sizes[0])
    g[i-1].fig.get_axes()[0].tick_params(axis='x', which='major', labelsize=font_sizes[0])
    if i==1 or i==7:  # beginning of a row
        g[i-1].fig.get_axes()[0].set_ylabel('DissimDiff', fontsize=font_sizes[2])
    else:
        g[i-1].fig.get_axes()[0].set_ylabel('')

fig = plt.figure(figsize=(20,11))
gs = gridspec.GridSpec(2, 6)

mg=[None]*12
for i,elem in enumerate(g):
    sfg.SeabornFig2Grid(elem, fig, gs[i])
gs.tight_layout(fig, h_pad=5, w_pad=2)
plt.show()

#%%
##############################
# Figure 4B ##################
##############################
dplt_allareas = res.groupby(id_factors).mean().\
        loc[:, np.r_[depvals]].reset_index()
dplt_allareas['Type_1'] = np.nan
dplt_allareas.loc[dplt_allareas.Type.str.startswith('pair33'), 'Type_1'] = 'pair33'
dplt_allareas.loc[dplt_allareas.Type.str.startswith('con50'), 'Type_1'] = 'con50'
dplt_allareas['Type_2'] = 'Label' 
dplt_allareas.loc[dplt_allareas.Type.str.endswith('sc'), 'Type_2'] = 'NoLabel' 
dplt_allareas['Type_1'] = dplt_allareas['Type_1'].replace(type_name_rplc_dict, regex=True)
dplt_allareas['DissimDiff'] = dplt_allareas['GDV_ur']-dplt_allareas['GDV_rel']
fig2 = plt.figure(figsize=(4,5))
fig2 = sns.barplot( data=dplt_allareas, x=hue_vs_x['x'], y='DissimDiff',
            palette=hue_vs_x_palette, order=x_order_list,
            hue=this_hue, ci=95,
            hue_order=hue_order_list )
fig2.legend_.set_title('')
fig2.set_xlabel('')
fig2.set_ylabel('')
fig2.set_ylim([0., 1.7])
fig2.get_figure().savefig('figs_highres_for_pub\\Fig4B.tiff',
             dpi=200)#, bbox_inches='tight')

#%%
##############################
# Figure 4C ##################
##############################
c2x2_palette = ['#ff7878', '#ffc8c8', '#7878ff', '#c8c8ff']
display_secndary = True
res['Sem'] = np.nan
res.loc[res.Type.str.startswith('pair33'), 'Sem'] = 'abs'
res.loc[res.Type.str.startswith('con50'), 'Sem'] = 'con'
res['Label'] = 'Label'
res.loc[res.Type.str.endswith('sc'), 'Label'] = 'NoLabel'
# Centrality2: centrality collapsed across visual/motor areas
res['Centrality2'] = np.nan
res.loc[res.AreaAbs.isin([1,6,]), 'Centrality2'] = 'PrimaryE'      # E: extrasylvian
res.loc[res.AreaAbs.isin([2,5,]), 'Centrality2'] = 'SecondaryE'    # E: extrasylvian
res.loc[res.AreaAbs.isin([3,4,]), 'Centrality2'] = 'CentralE'      # E: extrysylvian
res4 = res.reset_index().pivot_table(values=['GDV_ur','GDV_rel',], 
                      index = ['Sem','Centrality2','Model_ID'], columns='Label', aggfunc='mean')
res4[('DB','NoLBL')] = 0.0
res4[('DB','LBL')] = 100*(res4['GDV_ur']['Label']-res4['GDV_ur']['NoLabel'])/res4['GDV_ur']['NoLabel']
res4[('DW','NoLBL')] = 0.0
res4[('DW','LBL')] = 100*(res4['GDV_rel']['Label']-res4['GDV_rel']['NoLabel'])/res4['GDV_rel']['NoLabel']
res4 = res4.drop(columns=['GDV_ur','GDV_rel',])
res4.reset_index().groupby(['Sem','Centrality2','Model_ID']).mean()
res5=res4.reset_index().\
            melt(id_vars=[('Sem',''),('Centrality2',''),('Model_ID','')],
                value_vars=[('DB', 'LBL'), ('DW', 'LBL'),],
                var_name=['BW','Label'], value_name='PercDiff')
res5.columns = res5.columns.map(''.join)
res5['BW']=res5['BW'].replace({'DB':'between', 'DW':'within'})
res5['SemBW']=res5['BW']+' '+res5['Sem']
order_ = ['PrimaryE','CentralE']
if display_secndary:
    order_ = ['PrimaryE','SecondaryE','CentralE',]
ax = sns.pointplot(data=res5,                       
            y='PercDiff', x='Centrality2', order=order_,
            hue='SemBW',
            hue_order=['between abs','between con','within abs','within con'],
            ci=95, palette=np.array(c2x2_palette)[np.r_[0,2,1,3]])
ax.set_yticklabels(['%s%d%%'%("+" if x>0 else "", x) for x in ax.get_yticks()], fontsize=14)
ax.set_ylabel('')
ax.set_xticklabels(['Primary','Secondary','Central'],fontsize=12)
ax.set_xlabel('(extra-sylvian areas)', fontsize=11)
for t, l in zip(ax.legend_.texts, ['Dissim$_B$ abstract',
                                    'Dissim$_B$ concrete',
                                    'Dissim$_W$ abstract',
                                    'Dissim$_W$ concrete']):
    t.set_text(l)
ax.legend_.set_title('')
ax.set_xticklabels([x[:-1] for x in order_])