import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly as pltly
import plotly.express as px
import seaborn as sns
import scipy.stats as stats
import tap 

df = pd.read_csv(r"C:\Users\defrang\MGT_499\SOC perennials DATABASE.csv", skiprows=1, encoding='latin1')
print(df.head())
print(df.__len__())


#We want to aggregate the data by site, but to do so, we need to aggregate the data by plot and depth within each site

#We will filter the data to only include data from the top 20cm of soil. We will use the columns 'soil_from_cm_current', 'soil_to_cm_current'
#and 'soil_from_cm_previous' and 'soil_to_cm_previous' to determine the depth of the soil sample. We will then filter the data to only include samples from the top 20cm of soil.

#We will then aggregate the data by site, plot, and depth. We will calculate the mean SOC for each plot 


# Function to calculate weighted mean for current SOC values
def weighted_mean_current(group):
    total_depth = 0
    total_weighted_soc = 0
    for _, row in group.iterrows():
        depth_range = row['soil_to_cm_current'] - row['soil_from_cm_current']
        if row['soil_to_cm_current'] <= 20:  # Only consider depths up to 20 cm
            total_depth += depth_range
            total_weighted_soc += row['SOC_Mg_ha_current'] * depth_range
    return total_weighted_soc / total_depth if total_depth > 0 else None

# Function to calculate weighted mean for previous SOC values
def weighted_mean_previous(group):
    total_depth = 0
    total_weighted_soc = 0
    for _, row in group.iterrows():
        depth_range = row['soil_to_cm_previous'] - row['soil_from_cm_previous']
        if row['soil_to_cm_previous'] <= 20:  # Only consider depths up to 20 cm
            total_depth += depth_range
            total_weighted_soc += row['SOC_Mg_ha_previous'] * depth_range
    return total_weighted_soc / total_depth if total_depth > 0 else None

# Group by site and plot
grouped = df.groupby(['IDstudy', 'plotID'])

# Initialize lists to store results
results = []

for (site, plot), group in grouped:
    # Get mean SOC for the current values
    current_soc_mean = weighted_mean_current(group)

    # Get mean SOC for the previous values
    previous_soc_mean = weighted_mean_previous(group)

    # Append results for each site-plot combination
    results.append({'IDstudy': site, 'plotID': plot, 'country': group['country'].iloc[0],
                    'region': group['region'].iloc[0], 
                    'Mean_SOC_Mgha_Current': current_soc_mean,
                    'Mean_SOC_Mgha_Previous': previous_soc_mean})

# Convert results to a DataFrame
mean_soc_df = pd.DataFrame(results)

#Calculate the mean Mean_SOC_Mgha_Current and Mean_SOC_Mgha_Previous values for the entire dataset


print(mean_soc_df)

#Visualize the distribution of the mean SOC values for the current and previous data

#We will create a boxplot to visualize the distribution of the mean SOC values for the current and previous data

import seaborn as sns
import matplotlib.pyplot as plt

# Create a boxplot for the mean SOC values for the current and previous data
plt.figure(figsize=(10, 6))
sns.boxplot(data=mean_soc_df[['Mean_SOC_Mgha_Current', 'Mean_SOC_Mgha_Previous']])
plt.title('Distribution of Mean SOC Values')
plt.ylabel('Mean SOC (Mg/ha)')
plt.show()

#Calcuate the delta SOC values by subtracting the previous SOC values from the current SOC values

#We will create a new column in the mean_soc_df dataframe called 'Delta_SOC' that calculates the difference between the current and previous SOC values

# Calculate the delta SOC values
mean_soc_df['Delta_SOC'] = mean_soc_df['Mean_SOC_Mgha_Current'] - mean_soc_df['Mean_SOC_Mgha_Previous']

print(mean_soc_df)

#Visualize the distribution of the delta SOC values

#We will create a histogram to visualize the distribution of the delta SOC values

# Create a histogram for the delta SOC values
plt.figure(figsize=(10, 6))
sns.histplot(mean_soc_df['Delta_SOC'], kde=True)
plt.title('Distribution of Delta SOC Values')
plt.xlabel('Delta SOC (Mg/ha)')
plt.ylabel('Frequency')
plt.show()
