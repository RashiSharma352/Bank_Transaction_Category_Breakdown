import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# TITLE
st.title( "Bank Loan Data Analysis Dashboard")

# LOAD DATA
@st.cache_data
def load_data():
    df = pd.read_csv("synthetic_personal_finance_dataset.csv")
    return df

df = load_data()
# DATA PREVIEW

st.subheader("Dataset Preview")
st.dataframe(df.head())

# DATA INFO
st.subheader("Data Info")
st.write(df.info())

# FEATURE ENGINEERING
df['income_to_loan_ratio'] = df['monthly_income_usd'] / df['loan_amount_usd']

# SIDEBAR FILTER
st.sidebar.header(" Filter Data")

loan_type = st.sidebar.multiselect(
    "Select Loan Type",
    options=df['loan_type'].unique(),
    default=df['loan_type'].unique()
)

df_filtered = df[df['loan_type'].isin(loan_type)]

# KPI METRICS
st.subheader(" Key Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", len(df_filtered))
col2.metric("Avg Loan", round(df_filtered['loan_amount_usd'].mean(), 2))
col3.metric("Avg Income", round(df_filtered['monthly_income_usd'].mean(), 2))

# VISUALIZATIONS (OLD)
st.subheader("Basic Visualizations")

# Histogram
fig1, ax1 = plt.subplots()
sns.histplot(df_filtered['monthly_income_usd'], ax=ax1)
ax1.set_title("Income Distribution")
st.pyplot(fig1)

# Boxplot
fig2, ax2 = plt.subplots()
sns.boxplot(x=df_filtered['loan_amount_usd'], ax=ax2, color='orange')
ax2.set_title("Loan Amount Distribution")
st.pyplot(fig2)

# Scatter Plot
fig3, ax3 = plt.subplots()
sns.scatterplot(
    x=df_filtered['monthly_income_usd'],
    y=df_filtered['loan_amount_usd'],
    ax=ax3
)
ax3.set_title("Income vs Loan Amount")
st.pyplot(fig3)

# Heatmap
st.subheader("Correlation Heatmap")

fig4, ax4 = plt.subplots()
sns.heatmap(
    df_filtered.select_dtypes(include='number').corr(),
    annot=True,
    ax=ax4
)
st.pyplot(fig4)

# NEW VISUALIZATIONS (ADVANCED)
st.markdown("---")
st.subheader("Advanced Analysis")

#  Pie Chart - Loan Type
st.subheader("Loan Type Distribution")

loan_counts = df_filtered['loan_type'].value_counts()

fig5, ax5 = plt.subplots()
ax5.pie(loan_counts, labels=loan_counts.index, autopct='%1.1f%%')
ax5.set_title("Loan Type Distribution")
st.pyplot(fig5)

#  Bar Chart - Avg Loan per Type
st.subheader("Average Loan Amount by Loan Type")

avg_loan = df_filtered.groupby('loan_type')['loan_amount_usd'].mean()

fig6, ax6 = plt.subplots()
avg_loan.plot(kind='bar', ax=ax6)
ax6.set_ylabel("Avg Loan Amount")
plt.xticks(rotation=45)
st.pyplot(fig6)

#  Count Plot
st.subheader("Number of Customers per Loan Type")

fig7, ax7 = plt.subplots()
sns.countplot(x='loan_type', data=df_filtered, ax=ax7)
plt.xticks(rotation=45)
st.pyplot(fig7)

#  Income to Loan Ratio Distribution
st.subheader("Income to Loan Ratio Distribution")

fig8, ax8 = plt.subplots()
sns.histplot(df_filtered['income_to_loan_ratio'], kde=True, ax=ax8)
st.pyplot(fig8)

#  Horizontal Bar Chart
st.subheader("Top Loan Types")

top_loans = df_filtered['loan_type'].value_counts()

fig9, ax9 = plt.subplots()
top_loans.plot(kind='barh', ax=ax9)
st.pyplot(fig9)

#  Optional Default Pie Chart
if 'default' in df_filtered.columns:
    st.subheader("Default vs Non-Default")

    default_counts = df_filtered['default'].value_counts()

    fig10, ax10 = plt.subplots()
    ax10.pie(default_counts, labels=default_counts.index, autopct='%1.1f%%')
    st.pyplot(fig10)

# DOWNLOAD BUTTON
st.download_button(
    label=" Download Filtered Data",
    data=df_filtered.to_csv(index=False),
    file_name='filtered_data.csv',
    mime='text/csv'
)
# INSIGHTS
st.subheader(" Key Insights")

st.write(f"""
- Total Customers: {len(df_filtered)}
- Most Common Loan Type: {df_filtered['loan_type'].value_counts().idxmax()}
- Average Income: {round(df_filtered['monthly_income_usd'].mean(),2)}
- Average Loan: {round(df_filtered['loan_amount_usd'].mean(),2)}


Observations:
- Higher income customers tend to take higher loans  
- Some loan types dominate the dataset  
- Outliers are present in loan distribution  
- Income-to-loan ratio helps identify risky customers  
""")