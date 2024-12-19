#%%


import pandas as pd
from scipy.stats import chi2_contingency, spearmanr, pointbiserialr
import numpy as np

import plotnine as pn
#from plotnine import ggplot, aes, geom_density, facet_wrap, theme, labs, scale_color_manual, theme_light
from plotnine import *

#%%
# Read in results

#df = pd.read_csv('RESULTS/MODEL3/Final/Final_model3_predictions.txt', sep='\t')

df = pd.read_csv('ResultsNEWNEW/MODEL3/Final_model3_predictions.txt', sep='\t')

check_eval = pd.read_csv('ResultsNEWNEW/MODEL3/CV/cot_cnn_predictions.txt', sep='\t')



household = pd.read_csv('Data/household.csv')


household = household[household['countryname'] == 'Kenya']
# %%

count = 0
count2 = 0
households_predictions = pd.DataFrame()
for index, row in df.iterrows():
    print('Cluster ID:', row['clusterid'])  
    print('Number of Households', row['household_count_x'])
    print('Survey Year:', row['year_x'])



    print('Households in Survey:')
    matching_households = household[
        (household['clusterid'] == row['clusterid']) & 
        (household['year'] == row['year_x'])
    ]
    print(len(matching_households))

    matching_households = matching_households.assign(
    Image_Code=row['Image_Code'],
    Actual=row['Actual'],
    Predicted=row['Predicted']
    )

    households_predictions = pd.concat([households_predictions, matching_households], ignore_index=True)


    if len(matching_households) == row['household_count_x']:
        count += 1
    else:
        count2 += 1




    print('-------------')


# %%



df_melted = pd.melt(
    households_predictions,
    id_vars=['wealthindex'],
    value_vars=['Actual', 'Predicted'],
    var_name='Type',
    value_name='Category'
)

# Group data to calculate counts for bar heights
df_grouped = df_melted.groupby(['wealthindex', 'Type', 'Category']).size().reset_index(name='Count')
df_grouped['Type'] = pd.Categorical(df_grouped['Type'], categories=['Predicted', 'Actual'], ordered=True)

# Create the plot
from plotnine import ggplot, aes, geom_bar, facet_wrap, labs, theme, element_text, position_dodge

plot = (
    ggplot(df_grouped, aes(x='wealthindex', y='Count', fill='factor(Category)')) +
    geom_bar(stat='identity', position='dodge', width=0.7) +  # Ensure stat='identity' for pre-aggregated counts
    facet_wrap('~Type', ncol=2) +  # Facet for Actual and Predicted
    labs(
        title='Roof Quality by Wealth Index',
        x='Wealth Index',
        y='Count',
        fill='Roof Quality'
    ) +
    theme_minimal() +  # Apply minimal theme
    theme(
        axis_text_x=element_text(size=14),
        strip_text=element_text(size=20),
         plot_title=element_text(size=22),
        legend_position='top',
    ) +
    scale_fill_manual(
        values=["#a4041c", "#4974a5"],
        labels=["Class 0", "Class 1"] 
    )
)



plot


plot.save(
    filename = "ResultsNEWNEW/MODEL3/Visuals/roof_wealth.png", 
    dpi=300,  
    width=10,  
    height=6,  
    units="in"  
)
# %%



#TABLEs


for var in ['wealthindex',  'year', 'urbanrural', 'swatercat', 'watermins', 'electricity', 'cookfuel', 'edhigh', 'edlevel', 'floorcat', 'wallcat']:
    print('.........................')
    print('Variable:', var)
    cross_tab_actual = pd.crosstab(households_predictions['Actual'], households_predictions[var], margins=True)
    cross_tab_pred = pd.crosstab(households_predictions['Predicted'], households_predictions[var], margins=True)


    #print('Actual')
    print(cross_tab_actual)

    #print('Predicted')
    print(cross_tab_pred)

    print('.........................\n\n\n')
# %%


households_predictions['asset_count'] = households_predictions[['landline', 'mobile', 'watch', 'cart', 'motorboat', 
                                       'radio', 'tv', 'fridge', 'bicycle', 'scooter', 'car']].sum(axis=1)


households_predictions['asset_count_grouped'] = households_predictions['asset_count'].apply(lambda x: x if x < 5 else 5)

# Cross-tabulate grouped asset_count with roofcat
print('Actual')
print(pd.crosstab(households_predictions['Actual'], households_predictions['asset_count_grouped'], margins=True))

print('Predicted')
print(pd.crosstab(households_predictions['Predicted'], households_predictions['asset_count_grouped'], margins=True))


# %%
