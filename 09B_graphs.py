



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

# Correlations

households_predictions['Predicted'] = pd.to_numeric(households_predictions['Predicted'], errors='coerce')

households_predictions['urbanrural'] = households_predictions['urbanrural'].astype('category')
households_predictions['swatercat'] = households_predictions['swatercat'].astype('category')
households_predictions['tfacltycat'] = households_predictions['tfacltycat'].astype('category')
households_predictions['mobile'] = households_predictions['mobile'].astype('category')
households_predictions['hectare'] = households_predictions['hectare'].astype('category')
households_predictions['bankaccount'] = households_predictions['bankaccount'].astype('category')
households_predictions['electricity'] = households_predictions['electricity'].astype('category')
households_predictions['floorcat'] = households_predictions['floorcat'].astype('category')
households_predictions['wallcat'] = households_predictions['wallcat'].astype('category')
households_predictions['roofcat'] = households_predictions['roofcat'].astype('category')


households_predictions = households_predictions.dropna(subset=['Predicted'])


vars = ['urbanrural', 'swatercat', 'tfacltycat', 'bedrooms', 'mobile', 
        'hectare', 'agland', 'bankaccount', 'wealthindex', 'watermins',
        'electricity', 'ntoilet', 'edlevel', 'floorcat', 'wallcat', 'roofcat']

# Filter columns to avoid KeyErrors if any variable is missing
available_vars = [var for var in vars if var in households_predictions.columns]

# Results dictionaries
categorical_results = {}
numerical_results = {}

# Loop through variables to calculate correlation/association
for var in available_vars:
    if households_predictions[var].dtype.name == 'category' or households_predictions[var].dtype == 'object':  # Categorical variable
        if households_predictions[var].nunique() > 1:
            contingency_table = pd.crosstab(households_predictions[var], households_predictions['Predicted'])
            chi2, p, dof, expected = chi2_contingency(contingency_table)
            categorical_results[var] = {'Chi2': chi2, 'p-value': p}
    elif households_predictions[var].dtype in ['int64', 'float64']:  # Numerical variable
        if households_predictions[var].nunique() > 1:
            corr, p_val = spearmanr(households_predictions[var], households_predictions['Predicted'], nan_policy='omit')
            numerical_results[var] = {'Correlation': corr, 'p-value': p_val}

# Display Results
# Combine Categorical and Numerical Results
combined_results = []

for var, stats in categorical_results.items():
    combined_results.append({
        'Variable': var,
        'Statistic': f"Chi2: {stats['Chi2']:.4f}",
        'p-value': stats['p-value']
    })

# Add Numerical Results
for var, stats in numerical_results.items():
    combined_results.append({
        'Variable': var,
        'Statistic': f"Correlation: {stats['Correlation']:.4f}",
        'p-value': stats['p-value']
    })

def significance_level(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return ''

final_combined_df = pd.DataFrame(combined_results).sort_values(by='p-value')
final_combined_df['p-value'] = final_combined_df['p-value'].apply(lambda x: f"{x:.2e}" if not pd.isnull(x) else x)
final_combined_df['Significance'] = final_combined_df['p-value'].astype(float).apply(significance_level)

print(final_combined_df)


#%%
# VISUAL

households_predictions['Predicted'] = households_predictions['Predicted'].astype('category')
households_predictions['Actual'] = households_predictions['Actual'].astype('category')

# List of variables you want to plot
vars = ['urbanrural', 'swatercat', 'tfacltycat', 'bedrooms', 'mobile', 
        'hectare', 'agland', 'bankaccount', 'wealthindex', 'watermins',
        'electricity', 'ntoilet', 'edlevel', 'floorcat', 'wallcat']



vars = ['wealthindex']

df_melted = pd.melt(households_predictions, id_vars= 'Predicted', value_vars=vars)

plot = (ggplot(df_melted)
        + aes(x='value', fill='Predicted')  
        + geom_bar(stat='count', position='dodge', width=0.7)  
        + facet_wrap('~variable', scales='free_x', ncol=3)
        + theme_minimal()
        + theme(legend_position='top')
        + scale_fill_manual(values=["#a4041c", "#4974a5"], labels=["Low", "High"])
        + labs(x='', y='Count', fill='Roof Quality')
        +ggtitle('Predicted Roof Quality')+
        theme(panel_background=element_rect(fill='white'),
        axis_text_x=element_text(color='black', size=10),
        axis_text_y=element_text(color='black', size=10),
        axis_title_x=element_text(color='black', size=12, weight='bold'),
        axis_title_y=element_text(color='black', size=12, weight='bold'),
        strip_text=element_text(color='black', size=13, weight='bold'),
        legend_title=element_text(color='black', size=12, weight='bold'),
        legend_text=element_text(color='black', size=10),
        title=element_text(color='black', size=14, weight='bold'))

)



df_melted = pd.melt(households_predictions, id_vars= 'Actual', value_vars=vars)

plot2 = (ggplot(df_melted)
        + aes(x='value', fill='Actual')  
        + geom_bar(stat='count', position='dodge', width=0.7)  
        + facet_wrap('~variable', scales='free_x', ncol=3)
        + theme_minimal()
        + theme(legend_position='top')
        + scale_fill_manual(values=["#a4041c", "#4974a5"], labels=["Low", "High"])
        + labs(x='', y='Count', fill='Roof Quality')
        +ggtitle('Actual Roof Quality')+
        theme(panel_background=element_rect(fill='white'),
        axis_text_x=element_text(color='black', size=10),
        axis_text_y=element_text(color='black', size=10),
        axis_title_x=element_text(color='black', size=12, weight='bold'),
        axis_title_y=element_text(color='black', size=12, weight='bold'),
        strip_text=element_text(color='black', size=13, weight='bold'),
        legend_title=element_text(color='black', size=12, weight='bold'),
        legend_text=element_text(color='black', size=10),
        title=element_text(color='black', size=14, weight='bold'))

)













print(plot)

#%%
plot.save(
    'ResultsNEWNEW/MODEl3/Visuals/predictions_results.png',  # File path to save the plot
    width=12,  # Width of the plot in inches
    height=16,  # Height of the plot in inches
    dpi=300,  # Resolution (dots per inch)
    bbox_inches='tight',  # Ensures all plot elements are included
    transparent=False  # Use transparent=False for a solid background (True for transparent)
)




#%%




# Create a DataFrame


# Create contingency table
contingency_table = pd.crosstab(households_predictions["Predicted"], households_predictions["wealthindex"])

# Perform Chi-squared test
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Calculate Cramér's V
n = contingency_table.sum().sum()
cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))

print("Cramér's V:", cramers_v)

households_predictions = households_predictions.dropna(subset=["wealthindex"])



correlation, p_value = spearmanr(households_predictions["Predicted"], households_predictions["wealthindex"])


print(f"Spearman Correlation: {correlation}")
print(f"P-value: {p_value}")



#Change these to

#[nutrition , child s, 
# years of schooling, school attendance,
#   sanitation, drinking water, electricity,
#   cooking fuel, housing  assets]




#%%

variables_to_crosstab = ['wealthindex', 'urbanrural', 'swatercat', 'electricty']

# Create cross-tabulation for each variable
crosstab_tables = {}
for var in variables_to_crosstab:
    crosstab_tables[var] = pd.crosstab(
        [households_predictions['Actual'], households_predictions['Predicted']],
        households_predictions[var],
        rownames=['Actual', 'Predicted'],
        colnames=[var],
        margins=True  # Add totals
    )
    display(crosstab_tables)














#%%

"""






from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Prepare data for classification
X = households_predictions.drop(columns=['Predicted', 'Actual', 'Image_Code', 'longitude', 'latitude', 'hhweight', 'hhid', 'clusterid', 'roofcat',
'hhidcaseidentification'])
y = households_predictions['Predicted']

# Convert categorical variables to dummies
X = pd.get_dummies(X, drop_first=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Feature Importance
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print(feature_importances.head(10))

# Plot top 10 features
feature_importances.head(10).plot(kind='bar', figsize=(8, 6), title='Top Features Predicting Household Type')




# Pairplots


sns.pairplot(households_predictions, hue='Predicted', vars=['wealthscore', 'hhweight', 'hhmember'])

TODO FOR RESULTS


Density plots 
Chi square
Feature scores
Maybe pairplott or bar plot??
Box plots
COrrelations 

Do the same pre data (kenya)

Same on lcuster size 
Same on predictions

"""


# %%
