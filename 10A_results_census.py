#%%
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from adjustText import adjust_text
#%%

#df_full = pd.read_csv('RESULTS/MODEL3/Final/Final_model3_predictions.txt', sep='\t')

#df_full = pd.read_csv('Results_new/MODEL3/Final_model3_predictions.txt', sep='\t')
df_full = pd.read_csv('ResultsNEWNEW/MODEL3/Final_model3_predictions.txt', sep='\t')

household = pd.read_csv('Data/household.csv')

# %%



# Shap file

kenya_shapefile = gpd.read_file("Data/kenya_shape_files/County.shp")

# %%

df = df_full[['latitude', 'longitude', 'Predicted', 'Actual', 'lat', 'lon']]



geometry = [Point(xy) for xy in zip(df["longitude"], df["latitude"])]
points_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")


kenya_shapefile = kenya_shapefile.to_crs("EPSG:4326")


mapped_points = gpd.sjoin(points_gdf, kenya_shapefile, how="left", predicate="within")

custom_cmap = ListedColormap(["#a4041c", "#4974a5"]) 


fig, axs = plt.subplots(1, 2, figsize=(20, 10))

kenya_shapefile.plot(ax=axs[0], color="#EAE3DD", edgecolor="black")
mapped_points.plot(
    ax=axs[0],
    column="Predicted",  
    categorical=True,
    legend=True,
    markersize=50,
    cmap=custom_cmap 
)


for _, row in kenya_shapefile.iterrows():
    if row["COUNTY"] in ['Mandera', 'Wajir', 'Garissa', 'Samburur', 'Mombasa', 'Nyeri', 'Embu', 'Nyandarua']:
        axs[0].text(
            row.geometry.centroid.x + 0.1,  # X-coordinate of the centroid
            row.geometry.centroid.y -0.1,  # Y-coordinate of the centroid
            row["COUNTY"],  # Replace 'province' with the actual column name for province names
            fontsize=10,
            ha="center",
            va = 'center',
            color="black",
            bbox=dict(facecolor="#cdbcae", alpha=0.9, boxstyle='round', edgecolor="none")

    )
        
legend = axs[0].get_legend()
legend.set_title("Roof Quality",prop={"size": 20})
legend.get_texts()[0].set_text("Low") 
legend.get_texts()[1].set_text("High") 
legend.set_frame_on(False)  
axs[0].set_title("Predicted Roof Quality", fontsize=16)
axs[0].set_axis_off()




kenya_shapefile.plot(ax=axs[1], color="#EAE3DD", edgecolor="black")
mapped_points.plot(
    ax=axs[1],
    column="Actual",  
    categorical=True,
    legend=True,
    markersize=50,
    cmap=custom_cmap 
)


for _, row in kenya_shapefile.iterrows():
    if row["COUNTY"] in ['Mandera', 'Wajir', 'Garissa', 'Nairobi', 'Nyeri', 'Embu']:
        axs[1].text(
            row.geometry.centroid.x + 0.1,  # X-coordinate of the centroid
            row.geometry.centroid.y -0.1,  # Y-coordinate of the centroid
            row["COUNTY"],  # Replace 'province' with the actual column name for province names
            fontsize=10,
            ha="center",
            va = 'center',
            color="black",
            bbox=dict(facecolor="#cdbcae", alpha=0.9, boxstyle='round', edgecolor="none")

    )


legend = axs[1].get_legend()
legend.set_title("Roof Quality",prop={"size": 12})

legend.get_texts()[0].set_text("Low") 
legend.get_texts()[1].set_text("High") 
legend.set_frame_on(False)  

axs[1].set_title("Actual Roof Quality", fontsize=16)
axs[1].set_axis_off()


plt.tight_layout()
plt.savefig("ResultsNEWNEW/MODEL3/Visuals/combined_map.png", dpi=300, bbox_inches="tight", transparent=False)
plt.show()

# %%


# SCRAPE COUNTY STASTICS 


county_counts = (
    mapped_points.groupby(["COUNTY", "Predicted"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

# Rename columns for better readability
county_counts.columns = ["COUNTY", "Low Quality (0)", "High Quality (1)"]

# Display the counts
print(county_counts)
# %%

# SCRAPE COUNTY STATS 

urls = [
"https://en.wikipedia.org/wiki/List_of_counties_of_Kenya_by_GDP",
"https://en.wikipedia.org/wiki/Counties_of_Kenya",
"https://en.wikipedia.org/wiki/List_of_counties_of_Kenya_by_poverty_rate" ] 


for i, url in enumerate(urls, start=1):
        tables = pd.read_html(url)  
        print(f"URL {i}: Found {len(tables)} tables")
        for j, table in enumerate(tables, start=1):
            file_path = f"Data/kenya_shape_files/scrape/url_{i}_table_{j}.csv"
            table.to_csv(file_path, index=False)

# %%
# TABLES TO KEEP

table_files = [
    'Data/kenya_shape_files/scrape/url_1_table_3.csv',
    'Data/kenya_shape_files/scrape/url_1_table_4.csv',
    'Data/kenya_shape_files/scrape/url_2_table_3.csv',
    'Data/kenya_shape_files/scrape/url_3_table_1.csv'
]

county_stats = None

for file in table_files:
    

    df = pd.read_csv(file)
    df.rename(columns={'Name': 'county'}, inplace=True)
    df.columns = df.columns.str.lower()
    df['county'] = df['county'].replace({'Nairobi City': 'Nairobi', 'Nairobi County': 'Nairobi'})
    df['county'] = df['county'].replace({'Taita–Taveta': 'Taita Taveta', 'Taita-Taveta': 'Taita Taveta'})
    df['county'] = df['county'].replace({'Trans-Nzoia': 'Trans Nzoia'})

    if county_stats is None:
        county_stats = df
    else:
        county_stats = pd.merge(county_stats, df, on='county', how='outer')
# %%
# Merge predictions with county stats

county_counts['COUNTY'] = county_counts['COUNTY'].replace({'Keiyo-Marakwet' : 'Elgeyo-Marakwet', 'Tharaka' : 'Tharaka-Nithi'})

county_stats_cols = county_stats[['county','population (2019 census)',  'gcp per capita (usd)', 'value', '$2.15', '$3.65', '$6.85',     'former province']]

county_stats_merge = pd.merge(county_counts, county_stats_cols, left_on='COUNTY', right_on='county', how='left')



county_stats_merge['poverty_decimal'] = county_stats_merge['$2.15'].str.rstrip('%').astype(float) / 100





#%%%

county_counts_actual = (
    mapped_points.groupby(["COUNTY", "Actual"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

# Rename columns for better readability
county_counts_actual.columns = ["COUNTY", "ACTUAL Low Quality (0)", " ACTUAL High Quality (1)"]

county_counts_actual

county_stats_merge = pd.merge(county_stats_merge, county_counts_actual, on='COUNTY', how='left')


county_stats_merge.to_csv('Data/kenya_shape_files/county_stats_merged.csv',  index=False)





# %%


#%%
county_counts_actual = (
    mapped_points.groupby(["COUNTY", "Actual"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

# Rename columns for better readability
county_counts_actual.columns = ["COUNTY", "ACTUAL Low Quality (0)", " ACTUAL High Quality (1)"]

county_counts_actual




# %% 

#PLOT MAP WITH BIGGER AND CORRECT














# Define font sizes for consistency
title_font_size = 28
legend_title_font_size = 5
legend_text_font_size = 20
label_font_size = 16

fig, axs = plt.subplots(1, 2, figsize=(20, 10))

# Predicted Roof Quality Map
kenya_shapefile.plot(ax=axs[0], color="#fff0de", edgecolor="black")
mapped_points.plot(
    ax=axs[0],
    column="Predicted",
    categorical=True,
    legend=True,
    markersize=50,
    cmap=custom_cmap,
)

for _, row in kenya_shapefile.iterrows():
    if row["COUNTY"] in ['Mandera', 'Wajir', 'Garissa', 'Samburu', 'Mombasa', 'Nyeri', 'Embu', 'Nyandarua']:
        if row["COUNTY"] == "Nyandarua":
            axs[0].text(
                row.geometry.centroid.x - 0.35,
                row.geometry.centroid.y + 0.2,
                row["COUNTY"],
                fontsize=label_font_size, 
                ha="center",
                va="center",
                color="black",
                bbox=dict(facecolor="#cdbcae", alpha=0.9, boxstyle="round", edgecolor="none"),
            )
        
        else:
            axs[0].text(
            row.geometry.centroid.x + 0.1,
            row.geometry.centroid.y - 0.1,
            row["COUNTY"],
            fontsize=label_font_size,  # Adjusted label font size
            ha="center",
            va="center",
            color="black",
            bbox=dict(facecolor="#cdbcae", alpha=0.9, boxstyle="round", edgecolor="none"),
        )

legend = axs[0].get_legend()
legend.set_title("", prop={"size": legend_title_font_size})  # Adjust legend title font size
for text in legend.get_texts():
    text.set_fontsize(legend_text_font_size)  # Adjust legend text font size
legend.get_texts()[0].set_text("Predicted Class 0")
legend.get_texts()[1].set_text("Predicted Class 1")
legend.set_frame_on(False)
axs[0].set_title("Predicted Roof Quality", fontsize=title_font_size)
axs[0].set_axis_off()

# Actual Roof Quality Map
kenya_shapefile.plot(ax=axs[1], color="#fff0de", edgecolor="black")
mapped_points.plot(
    ax=axs[1],
    column="Actual",
    categorical=True,
    legend=True,
    markersize=50,
    cmap=custom_cmap,
)

for _, row in kenya_shapefile.iterrows():
    if row["COUNTY"] in ['Mandera', 'Wajir', 'Garissa', 'Samburu', 'Mombasa', 'Nyeri', 'Embu', 'Nyandarua']:
        if row["COUNTY"] == "Nyandarua":
            axs[1].text(
                row.geometry.centroid.x - 0.35,
                row.geometry.centroid.y + 0.2,
                row["COUNTY"],
                fontsize=label_font_size, 
                ha="center",
                va="center",
                color="black",
                bbox=dict(facecolor="#cdbcae", alpha=0.9, boxstyle="round", edgecolor="none"),
            )
        
        
        else:
            axs[1].text(
            row.geometry.centroid.x + 0.1,
            row.geometry.centroid.y - 0.1,
            row["COUNTY"],
            fontsize=label_font_size, 
            ha="center",
            va="center",
            color="black",
            bbox=dict(facecolor="#cdbcae", alpha=0.9, boxstyle="round", edgecolor="none"),
        )

legend = axs[1].get_legend()
legend.set_title("", prop={"size": legend_title_font_size})  # Adjust legend title font size
for text in legend.get_texts():
    text.set_fontsize(legend_text_font_size)  # Adjust legend text font size
legend.get_texts()[0].set_text("Actual Class 0")
legend.get_texts()[1].set_text("Actual Class 1")
legend.set_frame_on(False)
axs[1].set_title("Actual Roof Quality", fontsize=title_font_size)
axs[1].set_axis_off()

# Tight layout and save
plt.tight_layout()
plt.savefig("ResultsNEWNEW/MODEL3/Visuals/combined_map.png", dpi=300, bbox_inches="tight", transparent=False)
plt.show()
#%%




# %%
