import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from statsmodels.tsa.filters import hp_filter
import matplotlib.ticker as mtick

st.set_page_config(
    page_title="TD1 - Partie 1",
    page_icon="📈"
)

@st.cache_data
def load_data():
    df = pd.read_stata('./data/TD1/data.dta')
    df.drop(['countrycode', 'pop'], axis = 1, inplace = True)
    
    # df['year'] = pd.to_datetime(df['year'], format='%Y')
    # This doesn't work as %Y expect a 4-digit year...
    # df['year'] = pd.to_datetime(df['year'].astype('str').str.zfill(4), format='%Y')
    # Doesn't work either...
    
    return df

df = load_data()

st.title('TD1 - Partie 1')

# 1. PIB par tête
st.write("### 1. PIB par tête")

countries = ['France', 'United States', 'Argentina', 'Brazil', 'Japan', 'India', 'China']

countries_choice = st.multiselect(
    'Countries',
    options=df.country.unique(),
    default=countries
)

min_year, max_year = st.slider(
    'Period',
    min_value = 1,
    max_value = 2018,
    value=(1, 2018)
)

def plot_GDP_per_capita(df, countries, min_year, max_year):
    df = df[df['country'].isin(countries)]
    df = df[(df['year'] >= min_year) & (df['year'] <= max_year)]
    
    fig, ax = plt.subplots()
    
    sns.lineplot(data = df, x = 'year', y = 'gdppc', hue = 'country')
    
    # Style
    ax.set_title('GDP per capita over time')
    ax.set_xlabel('Year')
    ax.set_ylabel('GDP per capita')
    ax.grid(True)
    
    st.pyplot(fig)
    

plot_GDP_per_capita(df, countries_choice, min_year, max_year)

st.write("""
Les pays ont enregistré une augmentation significative de leur PIB par habitant depuis le 20ème siècle.

On peut distinguer deux groupes : les pays développés (les États-Unis, la France et le Japon) et les pays émergents (l'Inde, la Chine, l'Argentine et le Brésil). 

Il semble que l'écart relatif entre ces deux groupes reste constant dans le temps à partir du 20ème siècle.
""")

# 2. 3. 4. Évolution du PIB pour la France et les États-Unis
st.write("### 2. Croissance du PIB par tête")

df = df[df['country'].isin(['France', 'United States'])]
df = df[df['year'] >= 1820]
# GDP growth rate
df['gdppc_rate'] = df.groupby('country')['gdppc'].pct_change() * 100

min_year, max_year = st.slider(
    'Period',
    min_value = 1820,
    max_value = 2018,
    value=(1820, 2018)
)

def plot_GDPPC_growth(df, min_year, max_year):
    df = df[(df['year'] >= min_year) & (df['year'] <= max_year)]

    fig, ax = plt.subplots()

    sns.lineplot(data=df, x='year', y='gdppc_rate', hue='country', ax=ax)
    ax.set_title('Annual growth rate of GDP per capita')
    ax.set_ylabel('Growth rate (%)')
    ax.set_xlabel('Year')
    
    # Horizontal line at 0
    ax.axhline(0, color='black', linewidth=1)
    
    st.pyplot(fig)

plot_GDPPC_growth(df, min_year, max_year)

st.write("""
Les pays ont enregistré une augmentation significative de leur PIB par habitant depuis le 20ème siècle.

On peut distinguer deux groupes : les pays développés (les États-Unis, la France et le Japon) et les pays émergents (l'Inde, la Chine, l'Argentine et le Brésil). 

Il semble que l'écart relatif entre ces deux groupes reste constant dans le temps à partir du 20ème siècle.         
"""
)

# 3. Taux de croissance annuel moyen et écart type 

st.write("### 3. TCAM et écart type de la croissance annuelle du PIB par tête")

st.info(f"""
        **Average annual growth of GDP per capita between {min_year} and {max_year} :**
- France : {df[df['country'] == 'France']['gdppc_rate'].mean():.2f} %
- United States : {df[df['country'] == 'United States']['gdppc_rate'].mean():.2f} %
""")

st.info(f"""
        **Standard deviation of annual growth of GDP per capita between {min_year} and {max_year} :**
- France : {df[df['country'] == 'France']['gdppc_rate'].std():.2f} %
- United States : {df[df['country'] == 'United States']['gdppc_rate'].std():.2f} %
""")

# 4. Nombre de croissance annuelle négative du PIB par tête
st.write("### 4. Nombre de croissance annuelle négative du PIB par tête")
st.info(f"""
        **Number of negative annual growth of GDP per capita between {min_year} and {max_year} :**
- France : {df[df['country'] == 'France']['gdppc_rate'].lt(0).sum()}
- United States : {df[df['country'] == 'United States']['gdppc_rate'].lt(0).sum()}
""")

# 5. 6. Filtre HP
df3 = df.copy(deep=True)

# Set (country, year) as index
df3.set_index(['country', 'year'], inplace=True)

# Apply HP filter to GDPPC for each country
grouped = df3.groupby(level=0)['gdppc']

cycle, trend = zip(*grouped.apply(lambda x: hp_filter.hpfilter(x, lamb=6.25)))

df3['gdppc_trend'] = np.concatenate(trend)
df3['gdppc_cycle'] = np.concatenate(cycle)

# 7. Représentation graphique du filtre HP
st.write("### 7. Représentation graphique du filtre HP")

def plot_HP_filter(df3, country):
    temp = df3.loc[country]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(15, 8), gridspec_kw={'height_ratios': [2, 1]})
    
    # Plot GDP per capita and trend
    sns.lineplot(data=temp, x='year', y='gdppc', label='GDP per capita', ax=ax1)
    sns.lineplot(data=temp, x='year', y='gdppc_trend', label='Trend', ax=ax1)
    ax1.set_ylabel('GDP per capita')
    ax1.set_title(f'Trend and cycle of GDP per capita in {country}')
    ax1.grid(True)
    
    # Plot cycle
    sns.lineplot(data=temp, x='year', y='gdppc_cycle', label='Cycle (HP filters)', ax=ax2)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Cycle (deviation from trend)')
    ax2.grid(True)
    
    # Add secondary y-axis for cycle expressed as percentage of trend
    ax3 = ax2.secondary_yaxis('right', functions=(lambda x: x * 100 / temp['gdppc_trend'].max(), lambda x: x * temp['gdppc_trend'].max() / 100))
    ax3.set_ylabel('Deviation from trend (%)')
    ax3.set_ylim(-10, 10)
    ax3.yaxis.set_major_formatter(mtick.PercentFormatter())
    
    # Adjust layout and spacing
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.05)
    
    st.pyplot(fig)

country_choice = st.selectbox(
    'Country',
    options=['France', 'United States'],
    index=0
)

plot_HP_filter(df3, country_choice)

def plot_HP_filter_reg(country):
    temp = df3.loc[country]
    temp = temp.reset_index(level=0)    
    
    fig, ax = plt.subplots()
    # Plot trend of GDP per capita 
    sns.lineplot(data=temp, x='year', y='gdppc_trend', label='Trend')
    
    # Fit and plot linear regression
    sns.regplot(data=temp, x='year', y='gdppc_trend', label='Linear regression', order = 1, scatter = False, ci = 0)
    
    # Fit and plot polynomial regression
    sns.regplot(data=temp, x='year', y='gdppc_trend', label='Polynomial regression (degree = 2)', order=2, scatter = False, ci = 0)
    
    # Set title and labels
    plt.title(f'Linear and Polynomial Regression of Trend in {country}')
    plt.xlabel('Year')
    plt.ylabel('GDP per capita')
    
    # Add legend with regression coefficients
    plt.legend(title='Regression', loc='upper left', frameon=True, fancybox=True, shadow=True)
    
    st.pyplot(fig)
    
plot_HP_filter_reg(country_choice)

def plot5():
    """Plot polynomial regression of trend for France and United States aside"""
    fig, ax = plt.subplots()
    
    sns.regplot(data=df3.loc['France'].reset_index(level=0), x='year', y='gdppc_trend', label='France', order=2, scatter = False, ax = ax)
    sns.regplot(data=df3.loc['United States'].reset_index(level=0), x='year', y='gdppc_trend', label='United States', order=2, scatter = False, ax = ax)
    plt.suptitle(f'Regression of GPD per capita trend from 1825 to 2018')
    plt.title("(polynomial regression of order 2)")

    plt.xlabel('Year')
    plt.ylabel('GDP per capita')
    plt.legend(title='Country', loc='upper left', frameon=True, fancybox=True, shadow=True)
    
    st.pyplot(fig)
    
st.write('**Side-by-side comparison of trend regression :**')
plot5()


hide_streamlit_style = """
            <style>
            [data-testid="stToolbar"] {visibility: hidden !important;}
            footer {visibility: hidden !important;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


