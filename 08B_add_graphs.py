#%%
import pandas as pd
from scipy.stats import chi2_contingency, spearmanr, pointbiserialr
import numpy as np
import matplotlib.pyplot as plt

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

# DF AUC and Recall curve plots 

from scipy.interpolate import make_interp_spline
from sklearn.metrics import roc_curve

# Calculate ROC curve
fpr, tpr, _ = roc_curve(df['Actual'], df['Predicted_percent'])
precision, recall, thresholds = precision_recall_curve(df['Actual'], df['Predicted_percent'])

axis_font_size = 14
legend_font_size = 14
title_font_size = 18

# Function to remove duplicates (needed for smoothing)
def remove_duplicates(x, y):
    unique_indices = np.unique(x, return_index=True)[1]
    return x[unique_indices], y[unique_indices]

# Function to smooth curves using spline or fallback to linear interpolation
def robust_smooth_curve(x, y, points=200):
    x_new = np.linspace(x.min(), x.max(), points)
    if len(x) > 3:  # Use spline if enough points are available
        return x_new, make_interp_spline(x, y)(x_new)
    else:  # Fallback to linear interpolation for small datasets
        return x_new, interp1d(x, y, kind='linear')(x_new)

# Clean and smooth Precision-Recall data
recall_clean, precision_clean = remove_duplicates(np.array(recall), np.array(precision))
recall_smooth, precision_smooth = robust_smooth_curve(recall_clean, precision_clean)

# Clean and smooth ROC data
fpr_clean, tpr_clean = remove_duplicates(np.array(fpr), np.array(tpr))
fpr_smooth, tpr_smooth = robust_smooth_curve(fpr_clean, tpr_clean)

# Average Line for ROC Curve
average_tpr = np.linspace(0, 1, len(tpr_smooth))

# Define larger dimensions and font sizes to match the provided style
plt.figure(figsize=(14, 6))

# Define font sizes for axis labels, legend, and title
axis_font_size = 14
legend_font_size = 14
title_font_size = 18

# Precision-Recall Curve
plt.subplot(1, 2, 1)
plt.plot(recall_smooth, precision_smooth, color="#4974a5", linewidth=3)  # Increased linewidth
plt.xlabel("Recall", color="black", fontsize=axis_font_size)
plt.ylabel("Precision", color="black", fontsize=axis_font_size)
plt.title("Precision-Recall Curve", color="black", fontsize=title_font_size)

# ROC Curve
plt.subplot(1, 2, 2)
plt.plot(fpr_smooth, tpr_smooth, color="#a4041c", linewidth=3, label="ROC Curve")  # Increased linewidth
plt.plot(fpr_smooth, average_tpr, color="gray", linestyle="--", linewidth=2, label="Average TPR")  # Average line
plt.xlabel("False Positive Rate (FPR)", color="black", fontsize=axis_font_size)
plt.ylabel("True Positive Rate (TPR)", color="black", fontsize=axis_font_size)
plt.title("ROC Curve", color="black", fontsize=title_font_size)
plt.legend(loc='lower right', fontsize=legend_font_size)

# Layout and Save
plt.tight_layout()
plt.savefig("ResultsNEWNEW/MODEL3/Visuals/Validation2_curves.png")
plt.show()


# %%
