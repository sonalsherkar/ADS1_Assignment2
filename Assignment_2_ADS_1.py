# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 15:31:36 2023

@author: ss22alr
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import skew
from scipy.stats import kurtosis


def read_data(indicator_Name):
    """
       Read World Bank data from a CSV file, transpose the DataFrame,
       and drop the 'Country Code', 'Indicator Name', 'Indicator Code', and years columns.

       Parameters: - filename (str): The path to the CSV file containing World Bank data.
                   - df (pd.DataFrame): DataFrame with Country Name as column.
                   - transposed (pd.DataFrame): DataFrame with Year as row.
    """
    df = pd.read_csv("Data_from_World_Bank.csv", skiprows=3)

    # statistical analysis
    print("""\n  ------------------Statistical Methods-------------------  """)
    describe = df['2015'].describe()
    data = pd.DataFrame(describe)
    print(data)

    print("""\n  ------------------Mean and Median-------------------  """)
    mean_values = df['2015'].mean()
    median_values = df['2015'].median()
    # Prining mean and median values
    print("\nMean Values : ")
    print(mean_values)
    print("\nMedian Values : ")
    print(median_values)
    print("\n")
    
    countries = ['Romania', 'India', 'Mali', 'Singapore',
                 'United States', 'Aruba', 'Bahrain', 'Bangladesh', 'United Kingdom']
    d1 = df[(df['Indicator Name'] == indicator_Name)
            & (df['Country Name'].isin(countries))]

    d1_d = d1.drop(['Country Code', 'Indicator Name', 'Indicator Code', '1960', '1961', '1962', '1963', '1964', 
                    '1965', '1966', '1967', '1968', '1969', '1970','1971', '1972', '1973', '1974', '1975', '1976', 
                    '1977', '1978', '1979', '1980', '1981', '1982', '1983', '1984', '1985', '1986', '1987', '1988',
                    '1989', '1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', 
                    '2001', '2002', '2003', '2004', '2005', '2006','2007', '2008', '2009', '2010', '2011', '2012', 
                    '2013', '2022', 'Unnamed: 67'], axis=1).reset_index(drop=True)

    d1_t = d1_d.transpose()
    d1_t.columns = d1_t.iloc[0]
    d1_t = d1_t.iloc[1:]
    d1_t.index = pd.to_numeric(d1_t.index)
    d1_t['Years'] = d1_t.index
    d1_f = d1_t.reset_index(drop=True)
    return d1_d, d1_f


def slice_data(df1):
    """
        Here, we pass our dataset to this function, it return the data set with containing only required colunms.
    """
    df1 = df1[['Country Name', '2020']]
    return df1


def merge_four(r1, r2, r3, r4, r5, r6):
    """
        This merge function takes 6 arguments and returns us the one merged object file of all this 6 arguments by 
        considering the country_name as 'on' parameter.Here we also reset the index of merged object tovreset the 
        index back to default 0,1,2 etc indexes
    """
    merge1 = pd.merge(r1, r2, on='Country Name', how='outer')
    merge2 = pd.merge(merge1, r3, on='Country Name', how='outer')
    merge3 = pd.merge(merge2, r4, on='Country Name', how='outer')
    merge4 = pd.merge(merge3, r5, on='Country Name', how='outer')
    merge5 = pd.merge(merge4, r6, on='Country Name', how='outer')
    merge5 = merge5.reset_index(drop=True)
    return merge5


def Bar_plot_1(df):
    """
            Plots a Bar graph showing the Mortality rate, under-5(per 1,000 live births) percentage over the years 
            for selected countries.

            Parameters: df (pd.DataFrame): The DataFrame containing the data for the plot.
    """
    # Plotting bar graph 1
    plt.figure()
    df.plot(x='Years', y=['Romania', 'India', 'Mali', 'United Kingdom', 'Singapore', 'United States', 'Bangladesh'],
            kind='bar', xlabel='Years', ylabel='Total (%)')
    # Adding labels and title
    plt.legend(loc='upper right', fontsize='8',bbox_to_anchor=(1.30,1.0))
    plt.title('Bar Graph of Mortality rate, under-5(per 1,000 live births)')
    # Display a Output
    plt.grid(True)
    plt.show()


def Bar_plot_2(df):
    """
            Plots a Bar graph displays the Urban population (% of total population) over the years for selected 
            countries.

            Parameters: df (pd.DataFrame): The DataFrame containing the data for the plot.
    """
    # Plotting bar graph 1
    plt.figure()
    df.plot(x='Years', y=['Romania', 'India', 'Mali', 'United Kingdom', 'Singapore',
            'United States', 'Bangladesh'], kind='bar', xlabel='Years', ylabel='Total Percentages')
    # Adding labels and title
    plt.legend(loc='upper right', fontsize='8',bbox_to_anchor=(1.30,1.0))
    plt.title('Bar graph of Urban population (% of total population)')
    # Display a Output
    plt.grid(True)
    plt.show()


def Heat_map_1(df):
    """
            Creates a heatmap for the correlation matrix 1.

            Parameters: correlation_matrix (pd.DataFrame): The correlation matrix to visualize.
    """
    # Plots a Heat_map 1
    plt.figure(figsize=(11, 6))
    numeric_df = df.select_dtypes(include='number')
    correlation_matrix = numeric_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".3f", linewidths=.5)
    # Add a ticks and title
    plt.title('Correlation Matrix for Selected Indicators')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    # Display a Heat_map
    plt.show()


def Heat_map_2(df):
    """
            Creates a heatmap for the correlation matrix 2.

            Parameters: correlation_matrix (pd.DataFrame): The correlation matrix to visualize.
    """
    # Plots a Heat_map 2
    plt.figure(figsize=(11, 6))
    numeric_df = df.select_dtypes(include='number')
    correlation_matrix = numeric_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt=".2f", linewidths=.5)
    # Add a ticks and title
    plt.title('Correlation Matrix for Selected Indicators')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    # Display a Heat_map
    plt.show()


def Line_plot_1(df):
    """
            Plots a line graph displaying the Cereal yield Kg per Hectare for selected years and countries.

            Parameters: df (pd.DataFrame): The DataFrame containing the data for the plot.
    """
    # Plotting Line Plot 1
    plt.figure()
    df.plot(x='Years', y=['Aruba', 'India', 'Mali', 'Singapore', 'United States', 'Bangladesh',
            'United Kingdom'], kind='line', xlabel='Years', ylabel='kg per hectare', marker='o', linestyle='--')
    plt.legend(loc='upper right', fontsize='8',bbox_to_anchor=(1.30,1.0))
    # Display a title
    plt.title('Line Plot of Cereal yield')
    # Display a plot
    plt.grid(True)
    plt.show()


def Line_plot_2(df):
    """
            Plots a line graph shows the Renewable energy consumption(% of total final energy consumption) 
            for selected years and countries.

            Parameters: df (pd.DataFrame): The DataFrame containing the data for the plot.
    """
    # Plotting Line Plot 2
    plt.figure()
    df.plot(x='Years', y=['Bahrain', 'India', 'Mali', 'Singapore', 'United States', 'Aruba', 'Bangladesh',
            'United Kingdom'], kind='line', xlabel='Years', ylabel='Kg per hectare', marker='*', linestyle=':')
    plt.legend(loc='upper right', fontsize='8',bbox_to_anchor=(1.30,1.0))
    # Display a title
    plt.title('Line plot of Renewable energy consumption')
    # Display a plot
    plt.grid(True)
    plt.show()


def skew_and_kurtosis_plot(df, df1):
    """
            Calculate skewness & kurtosis and also Plots a histogram for the given data.

            Parameters: df, df1 : The data for which the histogram is to be plotted.  
    """
    # Calculate skewness and kurtosis
    numeric = pd.to_numeric(df, errors='coerce')
    numeric_value = numeric.dropna()
    skewness = stats.skew(numeric_value)
    kurtosis = stats.kurtosis(numeric_value)
    
    print("""\n------------------Skewness and Kurtosis-------------------\n""")
    print(f"Skewness : {skewness}")
    print(f"Kurtosis : {kurtosis}")


# It is indicators data for which the bar graph, line plot, and histogram is to be plotted
CY, Cereal_yield = read_data('Cereal yield (kg per hectare)')
MR, Mortality_rate = read_data('Mortality rate, under-5 (per 1,000 live births)')
UP, Urban_pop = read_data('Urban population (% of total population)')
AL, Agricultural_land = read_data('Agricultural land (sq. km)')
ele, electricity = read_data('Access to electricity (% of population)')
AGRI, agri = read_data('Agriculture, forestry, and fishing, value added (% of GDP)')

# here we slice the data and rename the ticks of x-axis and y-axis
CY_correlation = slice_data(CY).rename(columns={'2020': 'Cereal yield'})
MR_correlation = slice_data(MR).rename(columns={'2020': 'Mortality rate'})
UP_correlation = slice_data(UP).rename(columns={'2020': 'Urban population'})
AL_correlation = slice_data(AL).rename(columns={'2020': 'Agricultural land'})
ELE_correlation = slice_data(ele).rename(columns={'2020': 'Access to electricity'})
AGRI_correlation = slice_data(AGRI).rename(columns={'2020': 'Agriculture, forestry, & fishing'})

# Six indicator data is merge for Heat_map 1
df2 = merge_four(CY_correlation, MR_correlation, UP_correlation, AL_correlation, ELE_correlation, AGRI_correlation)

# It is indicators data for which the bar graph, line plot, and histogram is to be plotted
PG, pop_growth = read_data('Population growth (annual %)')
FOR, Foreign = read_data('Foreign direct investment, net inflows (% of GDP)')
EMI, emissions = read_data('CO2 emissions (kt)')
CON, consumption = read_data('Renewable energy consumption (% of total final energy consumption)')
SC, School = read_data('School enrollment, primary and secondary (gross), gender parity index (GPI)')
FA, forest = read_data('Forest area (sq. km)')

# here we slice the data and rename the ticks of x-axis and y-axis
PG_correlation = slice_data(PG).rename(columns={'2020': 'Population growth'})
FOR_correlation = slice_data(FOR).rename(columns={'2020': 'Foreign direct investment'})
EMI_correlation = slice_data(EMI).rename(columns={'2020': 'CO2 emissions'})
CON_correlation = slice_data(CON).rename(columns={'2020': 'Energy consumption'})
SC_correlation = slice_data(SC).rename(columns={'2020': 'School enrollment'})
FA_correlation = slice_data(FA).rename(columns={'2020': 'Forest area'})

# Six indicator data is merge for Heat_map 2
df3 = merge_four(PG_correlation, FOR_correlation, EMI_correlation, CON_correlation, SC_correlation, FA_correlation)

# Bar graph
Bar_plot_1(Mortality_rate)
Bar_plot_2(Foreign)

# Heat_map
Heat_map_1(df2)
Heat_map_2(df3)

# Line Graph
Line_plot_1(Cereal_yield)
Line_plot_2(consumption)

# Histogram with skewness and kurtosis
skew_and_kurtosis_plot(pop_growth['Aruba'], pop_growth['Singapore'])