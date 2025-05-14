import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import numpy as np
import streamlit_authenticator as stauth

st.set_page_config(layout="wide", page_title="Les Mills Member Analytics")

plt.style.use('dark_background')
plt.rcParams['axes.facecolor'] = '#0e1117'
plt.rcParams['figure.facecolor'] = '#0e1117'
plt.rcParams['savefig.facecolor'] = '#0e1117'

# --- Authentication Setup ---
names = ['Keyvan', 'Andrew']
usernames = ['ksalehi', 'Andrew']
passwords = ['lesmills', 'lesmills']
hashed_passwords =['$2b$12$vHzWaOhMVCc2xKi18XlMV.qMm.5.cv.qcEs7MmjDbBH6ZJiCAhNhS', '$2b$12$Pj3vvVu2TLGlqQvt5p7vW.5mZbvR4YvyaBeBYzSChsm.TUeDMM0mK']
authenticator = stauth.Authenticate(
    credentials={
        "usernames": {
            "Ksalehi": {
                "name": "Keyvan",
                "password": hashed_passwords[0]
            },
            "Andrew": {
                "name": "Andrew",
                "password": hashed_passwords[1]
            }
        }
    },
    cookie_name="lesmills_app",
    key="abc123",
    cookie_expiry_days=1
)

name, authentication_status, username = authenticator.login(location='main')



@st.cache_data
@st.cache_data
def load_main_data():
    try:
        server = 'LMNZLRPT001\\LM_RPT'
        database = 'LesMills_Reporting'
        driver = 'ODBC Driver 17 for SQL Server'
        conn_str = f"mssql+pyodbc://@{server}/{database}?driver={driver.replace(' ', '+')}&trusted_connection=yes"
        engine = create_engine(conn_str)
        query = "SELECT * FROM pbi.MXMData"
        return pd.read_sql_query(query, engine)
    except Exception as e:
        st.error(f"❌ Database connection failed: {e}")
        return pd.DataFrame()


@st.cache_data
def preprocess_main(df):
    df['CancellationReason'] = df['CancellationReason'].fillna('').astype(str).str.strip()
    df['AlertCloseTime'] = pd.to_datetime(df['AlertCloseTime'], errors='coerce')
    df['DateAdded'] = pd.to_datetime(df['DateAdded'], errors='coerce')
    df = df.dropna(thresh=int(0.5 * len(df)), axis=1)
    return df[df['CancellationReason'] != ''].copy()

@st.cache_data
def load_forecast_data():
    server = 'LMNZLRPT001\\LM_RPT'
    database = 'LesMills_Reporting'
    driver = 'ODBC Driver 17 for SQL Server'
    conn_str = f"mssql+pyodbc://@{server}/{database}?driver={driver.replace(' ', '+')}&trusted_connection=yes"
    engine = create_engine(conn_str)
    query = """
    SELECT 
        (SUM(s.SuspensionClosing) + SUM(d.JoinerClosing)) AS Membership_Base,
        SUM(ActiveMembershipClosing) AS Active_Membership,
        s.DateParameter
    FROM repo.MSReport_Summary_SuspensionsDTD AS s
    LEFT JOIN repo.MSReport_Summary_ActiveMembershipDetail AS d
        ON s.ClubID = d.ClubID AND s.DateParameter = d.DateParameter
    WHERE s.DateParameter > '2023-01-01'
    GROUP BY s.DateParameter
    ORDER BY s.DateParameter DESC
    """
    df = pd.read_sql_query(query, engine)
    df['DateParameter'] = pd.to_datetime(df['DateParameter'])
    return df.sort_values('DateParameter')

@st.cache_data
def load_join_leave_data():
    server = 'LMNZLRPT001\\LM_RPT'
    database = 'LesMills_Reporting'
    driver = 'ODBC Driver 17 for SQL Server'
    conn_str = f"mssql+pyodbc://@{server}/{database}?driver={driver.replace(' ', '+')}&trusted_connection=yes"
    engine = create_engine(conn_str)
    query = """
    SELECT 
        (SUM(Compleavers) + SUM(Leavers)) AS Leavers,
        (SUM(CompJoiners) + SUM(Joiners)) AS Joiners,
        DateParameter
    FROM repo.MSReport_Summary_ActiveMembershipDetail
    WHERE DateParameter > '2023-01-01'
    GROUP BY DateParameter
    ORDER BY DateParameter DESC
    """
    df = pd.read_sql_query(query, engine)
    df['DateParameter'] = pd.to_datetime(df['DateParameter'])
    return df.sort_values('DateParameter')

def show_eda(cancel_df):
    st.title("Cancellation Trends")
    top_reasons = cancel_df['CancellationReason'].value_counts().head(10).reset_index()
    top_reasons.columns = ['Reason', 'Count']

    fig, ax = plt.subplots()
    ax.barh(top_reasons['Reason'], top_reasons['Count'], color='#2ca02c')
    ax.set_title('Top 10 Cancellation Reasons')
    ax.set_xlabel('Count')
    ax.set_ylabel('Cancellation Reason')
    ax.invert_yaxis()
    st.pyplot(fig)

def show_ml(cancel_df):
    st.title("ML: Feature Importance")
    features = cancel_df[['Age', 'FeltValued', 'FeltWelcomed', 'LikelihoodToRejoin', 'EaseOfCancellation']].copy()
    for col in features.columns:
        features[col] = pd.to_numeric(features[col], errors='coerce')
    features = features.dropna()
    target = cancel_df.loc[features.index, 'CancellationReason']
    le = LabelEncoder()
    y = le.fit_transform(target)
    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.3, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    feat_imp = pd.Series(rf.feature_importances_, index=features.columns)
    feat_df = feat_imp.sort_values(ascending=False).reset_index()
    feat_df.columns = ['Feature', 'Importance']
    fig, ax = plt.subplots()
    ax.barh(feat_df['Feature'], feat_df['Importance'], color='#1f77b4')
    ax.set_title('Feature Importance')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    st.pyplot(fig)

def show_forecast():
    st.title("Forecast Membership Metrics")
    df = load_forecast_data()
    recent = df[df['DateParameter'] >= df['DateParameter'].max() - pd.Timedelta(days=7)]
    fig, ax = plt.subplots()
    ax.plot(recent['DateParameter'], recent['Membership_Base'], label='Membership Base', marker='o', color='#2ca02c')
    ax.plot(recent['DateParameter'], recent['Active_Membership'], label='Active Membership', marker='x', color='#1f77b4')
    ax.set_title('Membership Base and Active Members Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Member Count')
    ax.legend()
    st.pyplot(fig)
    df['Week'] = range(len(df))
    for col in ['Membership_Base', 'Active_Membership']:
        st.subheader(f"\ud83d\udd2e Forecast for {col}")
        model = LinearRegression()
        model.fit(df[['Week']], df[col])
        future_weeks = np.array(range(len(df), len(df) + 6)).reshape(-1, 1)
        future_dates = pd.date_range(df['DateParameter'].max(), periods=6, freq='W')
        forecast = model.predict(future_weeks)
        forecast_df = pd.DataFrame({'Date': future_dates.date, f'Predicted {col}': forecast.astype(int)})
        st.dataframe(forecast_df.set_index('Date'))

def show_join_leave_page():
    st.title("Joiners vs Leavers Analysis")
    df = load_join_leave_data()
    recent = df[df['DateParameter'] >= df['DateParameter'].max() - pd.Timedelta(days=7)]
    fig, ax = plt.subplots()
    ax.plot(recent['DateParameter'], recent['Joiners'], label='Joiners', marker='o', color='green')
    ax.plot(recent['DateParameter'], recent['Leavers'], label='Leavers', marker='x', color='red')
    ax.set_title('Joiners vs Leavers Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Member Count')
    ax.legend()
    st.pyplot(fig)

    df['Week'] = range(len(df))
    for col in ['Joiners', 'Leavers']:
        st.subheader(f"Forecast for {col}")
        model = LinearRegression()
        model.fit(df[['Week']], df[col])
        future_weeks = np.array(range(len(df), len(df) + 6)).reshape(-1, 1)
        future_dates = pd.date_range(df['DateParameter'].max(), periods=6, freq='W')
        forecast = model.predict(future_weeks)
        forecast_df = pd.DataFrame({'Date': future_dates.date, f'Predicted {col}': forecast.astype(int)})
        fig, ax = plt.subplots()
        ax.plot(df['DateParameter'], df[col], label='Actual', color='#2ca02c')
        ax.plot(forecast_df['Date'], forecast_df[f'Predicted {col}'], label='Forecast', linestyle='--', color='#ff7f0e')
        ax.set_title(f"{col} Forecast")
        ax.set_xlabel('Date')
        ax.set_ylabel(col)
        ax.legend()
        st.pyplot(fig)
        st.dataframe(forecast_df.set_index('Date'))

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

@st.cache_data
def load_cluster_data():
    server = 'LMNZLRPT001\\LM_RPT'
    database = 'LesMills_Reporting'
    driver = 'ODBC Driver 17 for SQL Server'
    conn_str = f"mssql+pyodbc://@{server}/{database}?driver={driver.replace(' ', '+')}&trusted_connection=yes"
    engine = create_engine(conn_str)
    query = """
    WITH RankedData AS (
        SELECT 
            a.LesMillsID,
            a.MemberFirstName,
            a.MemberLastName,
            a.Status,
            a.AttendanceInLast90Days,
            a.emailAddress,
            a.AttendanceBracket,
            a.MembershipFeeType,
            a.DiscountAmount,
            a.PaymentFrequency,
            a.RegularPayment,
            s.Age,
            s.likelihoodToRecommend,
            ROW_NUMBER() OVER (PARTITION BY s.MemberId ORDER BY s.AlertCloseTime DESC) AS rn
        FROM pbi.MXMData AS s
        INNER JOIN repo.ActiveMemberReport AS a
            ON s.MemberId = a.LesMillsID
    )
    SELECT * FROM RankedData WHERE rn = 1
    """
    df = pd.read_sql_query(query, engine)
    return df

def show_cluster_page():
    st.title("Customer Clustering")
    df = load_cluster_data()
    cluster_data = df[['Age', 'RegularPayment', 'AttendanceInLast90Days', 'likelihoodToRecommend']].copy()
    for col in cluster_data.columns:
        cluster_data[col] = pd.to_numeric(cluster_data[col], errors='coerce')
    cluster_data = cluster_data.dropna()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(cluster_data)

    k = st.slider("Select number of clusters", 2, 10, 4)
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(scaled)
    cluster_data['Cluster'] = labels
    pca = PCA(n_components=2)
    components = pca.fit_transform(scaled)

    fig, ax = plt.subplots()
    cluster_names = {
        0: "Loyal High Spender",
        1: "At Risk Low Attendance",
        2: "New Member Moderate Activity",
        3: "Infrequent Visitor"
    }
    for cluster_num in sorted(set(labels)):
        idx = cluster_data['Cluster'] == cluster_num
        ax.scatter(
            components[idx, 0], components[idx, 1], 
            label=cluster_names.get(cluster_num, f"Cluster {cluster_num}"), alpha=0.7
        )
    ax.set_title("2D Cluster Plot by Cluster")
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.legend(title="Cluster")
    st.pyplot(fig)

    df = df.loc[cluster_data.index].copy()
    df['Cluster'] = labels
    st.subheader("Search Member by LesMillsID")
    member_id = st.text_input("Enter LesMillsID")
    if member_id:
        match = df[df['LesMillsID'].astype(str) == member_id]
        if not match.empty:
            cluster_id = match.iloc[0]['Cluster']
            cluster_label = cluster_names.get(cluster_id, f"Cluster {cluster_id}")
            st.success(f"Cluster: {cluster_label}")
            st.info(f"Description: {cluster_label} - This customer is likely {['very loyal and active with high spending', 'at risk due to low activity', 'moderately engaged and new', 'inactive and potentially churned'][cluster_id]}")
            st.dataframe(match)
        else:
            st.warning("Member ID not found.")

    st.subheader("Cluster Summary")
    numeric_cols = ['Age', 'RegularPayment', 'AttendanceInLast90Days', 'likelihoodToRecommend']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    cluster_summary = df.groupby('Cluster')[numeric_cols].mean().reset_index()
    cluster_summary['Cluster Name'] = cluster_summary['Cluster'].map(cluster_names)
    cluster_summary['Description'] = cluster_summary['Cluster'].map({
        0: "Very loyal and active with high spending",
        1: "At risk due to low activity",
        2: "Moderately engaged and new",
        3: "Inactive and potentially churned"
    })
    st.dataframe(cluster_summary[['Cluster', 'Cluster Name', 'Description'] + numeric_cols])

def main():
    
    st.sidebar.image("res.jpg", width=200)
    page = st.sidebar.radio("Navigation", [
        "EDA View", 
        "ML Feature Importance", 
        "Membership Forecast", 
        "Joiners vs Leavers",
        "Customer Clustering"
    ])
    df = load_main_data()
    cancel_df = preprocess_main(df)
    if page == "EDA View":
        show_eda(cancel_df)
    elif page == "ML Feature Importance":
        show_ml(cancel_df)
    elif page == "Membership Forecast":
        show_forecast()
    elif page == "Joiners vs Leavers":
        show_join_leave_page()
    elif page == "Customer Clustering":
        show_cluster_page()

if authentication_status == False:
    st.error("\u274c Incorrect username or password")
elif authentication_status == None:
    st.warning("\u26a0\ufe0f Please enter your credentials")
elif authentication_status:
    authenticator.logout('Logout', 'sidebar')
    st.success(f"\u2705 Welcome, {name}")
    main()
