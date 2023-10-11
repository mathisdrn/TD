import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import matplotlib.dates as mdates
import matplotlib.ticker as mtick

st.set_page_config(
    page_title="TD1 - Partie 2",
    page_icon="📈"
)

@st.cache_data
def load_data():
    df = pd.read_csv('./data/TD1/OECD - dataset.csv')
    df['Année'] = pd.to_datetime(df['TIME'], format='%Y')

    # Rename Variable column 
    variable_map = {
        'Formation brute de capital, total, en volume' : 'FBC',
        'Inflation globale harmonisée' : 'Inflation',
        'Produit intérieur brut, en volume, aux prix du marché' : 'PIB',
        'Solde financier primaire des administrations publiques, en pourcentage du PIB' : 'SFPAP',
        'Dépense de consommation finale du secteur privé, en volume' : 'DCFSP'
    }
    df['Variable'] = df['Variable'].map(variable_map)

    df = df[['Pays', 'Année', 'Variable', 'Value']]
    
    return df

df = load_data()

st.title('TD1 - Partie 2 : Les fluctuations')

# 1. Soldes financiers
st.write("### 1. Soldes financiers")

def plot6(df):
    fig, ax = plt.subplots()
    
    # Plot SFPAP
    sns.lineplot(data=df[df.Variable == 'SFPAP'], x='Année', y='Value', hue = 'Pays')

    # Set title and labels
    plt.title(f'Solde financier primaire des administrations publiques, en pourcentage du PIB')
    plt.xlabel('Année')
    plt.ylabel('Pourcentage du PIB')
    # Mtick formatter
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())

    # Format date axis to show only year
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    # Add grid
    # plt.grid(True)
    plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
    
    # Make legend outside of plot
    plt.legend(title='Pays', loc='upper left', bbox_to_anchor=(1.05, 1), frameon=True, fancybox=True, shadow=True)
    # Make plot larger
    plt.gcf().set_size_inches(15, 5)
    
    # Add redline at 0%
    plt.axhline(y=0, color='r', linestyle='--')
    
    st.pyplot(fig)
    
plot6(df)

st.write("Les États-Unis, le Royaume-Uni, la Grèce et la France fournissent un effort budgétaire important (déficit publique primaire en moyenne de 2% entre 2000 et 2023).")
st.write("On observe une forte augmentation du déficit primaire des administrations publiques lors de la crise du Covid en 2020 et de la crise économique de 2008.")

# 2. Croissance du PIB
st.write("### 2. Croissance du PIB")

pib_tcam_temp = df.copy()
# Select Variable PIB, year in 2000 or 2023
pib_tcam_temp = pib_tcam_temp[(pib_tcam_temp.Variable == 'PIB') & ((pib_tcam_temp.Année == '2000') | (pib_tcam_temp.Année == '2023'))]
# For each country compute CAGR
pib_tcam_temp = pib_tcam_temp.pivot(index = 'Pays', columns = 'Année', values = 'Value')
pib_tcam_temp['TCAM'] = (pib_tcam_temp[pib_tcam_temp.columns[pib_tcam_temp.columns.year == 2023][0]] / pib_tcam_temp[pib_tcam_temp.columns[pib_tcam_temp.columns.year == 2000][0]]) ** (1 / 23) - 1
pib_tcam_temp['Taux de croissance annuel moyen'] = (round(pib_tcam_temp['TCAM'] * 100, 2)).astype(str) + " %"

st.write('Taux de croissance annuel moyen du PIB entre 2000 et 2023')
st.table(pib_tcam_temp['Taux de croissance annuel moyen'].sort_values(ascending = False))

def plot7(df):
    temp = df.copy()
    temp = temp[temp.Variable == 'PIB']
    # Compute growth rate
    temp['growth_rate'] = temp.groupby('Pays')['Value'].pct_change() * 100
    
    fig, ax = plt.subplots()
    
    sns.lineplot(data=temp, x='Année', y='growth_rate', hue='Pays', ax = ax)
    plt.axhline(y=0, color='r', linestyle='--')
    
    plt.title('GDP growth rate between 2000 and 2023')
    plt.xlabel('Year')
    plt.ylabel('Growth rate')
    plt.grid(True)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gcf().set_size_inches(17, 10)
    plt.legend(title='Country', loc='upper left', bbox_to_anchor=(1.05, 1), frameon=True, fancybox=True, shadow=True)
    plt.show()
    
    st.pyplot(fig)
    
plot7(df)

# 4. Solde financier moyen
st.write("### 4. Solde financier moyen")

st.write("**Solde financier primaire moyen des APU en pourcentage du PIB entre 2000 et 2023 :**")

avg_temp = df.copy()
avg_temp = avg_temp[avg_temp.Variable == 'SFPAP']
avg_temp = avg_temp.groupby('Pays')['Value'].mean()
avg_temp = avg_temp.sort_values(ascending = False)
avg_temp = (round(avg_temp, 2)).astype(str) + " %"

st.table(avg_temp)

# Comparaison avec le TCAM du PIB
st.write("**Comparaison avec le TCAM du PIB**")

def plot_comparison(df):
    # Solde financier moyen
    temp = df.copy()
    temp = temp[temp.Variable == 'SFPAP']
    temp = temp.groupby('Pays')['Value'].mean()
    temp = temp.sort_values(ascending = False)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'hspace': 0.1})

    # Barchart of average SFPAP
    sns.barplot(x=temp.index, y=temp.values, ax=ax1)
    ax1.set_title('Average SFPAP between 2000 and 2023')
    ax1.set_xlabel('')
    ax1.set_ylabel('SFPAP')
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter())

    # Barchart of GDP GACR
    sns.barplot(x=pib_tcam_temp.index, y=pib_tcam_temp['TCAM'], ax=ax2)
    ax2.set_title('GDP GACR between 2000 and 2023')
    ax2.set_xlabel('Pays')
    ax2.set_ylabel('TCAM')
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())

    st.pyplot(fig)
    
plot_comparison(df)

# 5. Part de la dépense privée et de l'investissement dans le PIB
st.write("### 5. Part de la dépense privée et de l'investissement dans le PIB")

def plot8(df):
    temp = df.copy()
    temp = temp[temp.Variable.isin(['FBC', 'PIB'])]
    # For each country and year compute FBCF / PIB
    temp = temp.pivot(index = ['Pays', 'Année'], columns = 'Variable', values = 'Value')
    temp['FBCF / PIB'] = (temp['FBC'] / temp['PIB']) * 100
    temp = temp.reset_index(level=1)

    fig, ax = plt.subplots()

    sns.lineplot(data=temp, x='Année', y='FBCF / PIB', hue = 'Pays', ax = ax)
    
    plt.ylabel('Pourcentage du PIB')
    
    plt.title('Part de l’investissement en % du PIB')
    plt.legend(title='Country', loc='upper left', bbox_to_anchor=(1.05, 1), frameon=True, fancybox=True, shadow=True)
    
    # Style
    plt.grid(True)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gcf().set_size_inches(15, 5)
    
    st.pyplot(fig)
    
    
plot8(df)

def plot11():
    temp = df.copy()
    temp = temp[temp.Variable.isin(['DCFSP', 'PIB'])]
    # For each country and year compute FBCF / PIB
    temp = temp.pivot(index = ['Pays', 'Année'], columns = 'Variable', values = 'Value')
    temp['DCFSP / PIB'] = (temp['DCFSP'] / temp['PIB']) * 100
    
    temp = temp.reset_index(level=1)

    fig, ax = plt.subplots()
    sns.lineplot(data=temp, x='Année', y='DCFSP / PIB', hue = 'Pays', ax = ax)
    
    plt.ylabel('Pourcentage du PIB')
    
    plt.title('Part de la dépense privée en % du PIB')
    plt.legend(title='Country', loc='upper left', bbox_to_anchor=(1.05, 1), frameon=True, fancybox=True, shadow=True)
    
    # Style
    plt.grid(True)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gcf().set_size_inches(15, 5)

    st.pyplot(fig)
    
plot11()

st.write("""

La formation brute de capital correspond à l'ensemble des dépenses d'investissement réalisées par les entreprises. 

Rapportée au % du PIB elle exprime la part de l'investissement dans l'économie.

À partir de 2010, il semble se dégager une hiérarchie dans la part des dépenses de l'investissement des pays :

Dans l'ordre décroissant : France, (Allemagne, Zone Euro, États-Unis), Royaume-Uni, Grèce.         
""")

# 6. Taux de croissance de la dépense privée et de l’investissement
st.write("### 6. Taux de croissance de la dépense privée et de l’investissement")

def plot9(df): 
    temp = df.copy()
    temp = temp[temp.Variable.isin(['DCFSP', 'FBC'])]
    
    temp = temp.pivot(index = ['Pays', 'Année'], columns = 'Variable', values = 'Value')
    temp['GSCF'] = temp['DCFSP'] + temp['FBC']
        
    # GSCF Growth rate
    temp['GSCF_growth_rate'] = temp.groupby('Pays')['GSCF'].pct_change() * 100
    
    
    fig, ax = plt.subplots()    
    sns.lineplot(data=temp, x='Année', y='GSCF_growth_rate', hue = 'Pays')
    plt.axhline(y=0, color='r', linestyle='--')

    plt.title('Growth rate of Gross Fixed Capital Formation')
    plt.xlabel('Year')
    plt.ylabel('Growth rate')
    plt.legend(title='Country', loc='upper left', bbox_to_anchor=(1.05, 1), frameon=True, fancybox=True, shadow=True)
    plt.grid(True)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gcf().set_size_inches(12, 8)
    
    st.pyplot(fig)
    
plot9(df)

def plot10(df):
    temp = df.copy()
    temp = temp[temp.Variable.isin(['DCFSP', 'FBC'])]
    
    temp = temp.pivot(index = ['Pays', 'Année'], columns = 'Variable', values = 'Value')
    temp['GSCF'] = temp['DCFSP'] + temp['FBC']
        
    # GSCF Growth rate
    temp['GSCF_growth_rate'] = temp.groupby('Pays')['GSCF'].pct_change() * 100    # Compute standard deviation for each country
    # STD for GSCF growth rate
    temp = temp.groupby('Pays')['GSCF_growth_rate'].std().sort_values(ascending = False)
    # Normalize :
    temp = temp / temp.max()

    fig, ax = plt.subplots()
    sns.barplot(x = temp.index, y = temp.values)
    # Adjust title and labels
    plt.title('Normalized standard deviation of growth rate of gross fixed capital formation')
    plt.xlabel('Country')
    plt.ylabel('Standard deviation (normalized)')
    # Style
    plt.gcf().set_size_inches(10, 5)
    
    st.pyplot(fig)
    
plot10(df)


st.write("""
La Grèce est le pays ayant connu les plus forte variation de dépenses privées et d'investissement entre 2000 et 2023. 

Les États-Unis, la France et l'Allemagne ont connu des variations de dépenses privées et d'investissement plus faibles.
""")


hide_streamlit_style = """
            <style>
            [data-testid="stToolbar"] {visibility: hidden !important;}
            footer {visibility: hidden !important;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


