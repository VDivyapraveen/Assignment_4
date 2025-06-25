import streamlit as st
import pandas as pd
import numpy as np
import base64
from sqlalchemy import create_engine
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# --- VERY FIRST STREAMLIT COMMAND ---
st.set_page_config(layout="wide")
st.markdown("""
    <style>
    .stApp {
        background-color: transparent;
    }
    section[data-testid="stSidebar"] {
        background: transparent !important;
        border-right: 3px solid black !important;
    }
    .block-container .stSidebarContent {
        padding-top: 2rem;
        background: rgba(255, 255, 255, 0.4);
        border-radius: 12px;
    }
    .sidebar-container {
        background-color: rgba(255, 255, 255, 0.4);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    .sidebar-title {
        font-size: 26px;
        font-weight: bold;
        color: black;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
    }
    .sidebar-radio > div > label {
        background-color: #b36920 !important;
        color: white !important;
        font-weight: bold !important;
        border-radius: 6px;
        padding: 8px 12px;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
    }
    .sidebar-radio > div > label:hover {
        background-color: #8f4e16 !important;
    }
    .sidebar-radio > div > label > div:first-child {
        margin-right: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Background Image Setup ---
def set_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: linear-gradient(rgba(255, 255, 255, 0.5), rgba(255, 255, 255, 0.5)), 
                          url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_bg_from_local(r"D:\\MyProjectsDuplicates\\project4\\logo_make_11_06_2023_175.jpg")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans&display=swap');
    * {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-weight: bold !important;
    }
    html, body, [class*="css"], [class^="st-"], [class*="st-"] {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-weight: bold !important;
    }
    .stMarkdown, .stDataFrame, .stTable, .stTextInput, .stSelectbox, .stSlider, .stNumberInput,
    .stButton, .stRadio, .stMultiSelect, .stDateInput, .stCheckbox {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-weight: bold !important;   
    }
    </style>
""", unsafe_allow_html=True)

# --- MAIN MENU TITLE IN SIDEBAR ---
with st.sidebar:
    st.markdown('<div class="sidebar-title">㊂ MAIN MENU</div><hr>', unsafe_allow_html=True)

# --- Load Data and Models ---
@st.cache_data
def load_data():
    engine = create_engine("mysql+pymysql://root:@localhost:3306/project4")
    return pd.read_sql("SELECT * FROM data", con=engine)

@st.cache_resource
def load_models():
    rating_model = joblib.load(r"D:\\MyProjectsDuplicates\\project4\\lightgbm_rating_model.pkl")
    encoders = joblib.load(r"D:\\MyProjectsDuplicates\\project4\\all_label_encoders.pkl")
    clf_pipeline = joblib.load(r"D:\\MyProjectsDuplicates\\project4\\visit_mode_classifier.pkl")
    return rating_model, encoders, clf_pipeline

@st.cache_resource
def train_recommender(df):
    matrix = df.pivot_table(index='UserId', columns='AttractionId', values='Rating')
    centered = matrix.sub(matrix.mean(axis=1), axis=0).fillna(0)
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(centered)
    return knn, matrix, centered

# --- Utility Functions ---
def get_season(month):
    return ('Winter' if month in [12, 1, 2] else 'Spring' if month in [3, 4, 5] else 'Summer' if month in [6, 7, 8] else 'Fall')

def recommend_attractions(user_id, knn, matrix, centered, df, top_n=5):
    if user_id not in centered.index:
        return None, f"User ID {user_id} not found."
    vec = centered.loc[user_id].values.reshape(1, -1)
    _, idx = knn.kneighbors(vec, n_neighbors=top_n+1)
    similar_users = centered.index[idx.flatten()[1:]]
    sim_ratings = matrix.loc[similar_users]
    mean_ratings = sim_ratings.mean().sort_values(ascending=False)
    visited = matrix.loc[user_id].dropna().index
    recs = mean_ratings.drop(visited, errors='ignore').head(top_n)
    names = df[['AttractionId', 'Attraction']].drop_duplicates().set_index('AttractionId')
    return pd.DataFrame({'AttractionId': recs.index}).join(names, on='AttractionId').reset_index(drop=True), None

# --- Helper Function for Plot Styling ---
def update_transparent(fig):
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(
            family='IBM Plex Sans',
            size=16,
            color='black'
        ),
        title_font=dict(
            size=25,
            color='black',
            family='IBM Plex Sans'
        )
    )
    fig.update_xaxes(title_font=dict(size=16, color='black', family='IBM Plex Sans'), 
                     tickfont=dict(size=14, color='black', family='IBM Plex Sans'))
    fig.update_yaxes(title_font=dict(size=16, color='black', family='IBM Plex Sans'), 
                     tickfont=dict(size=14, color='black', family='IBM Plex Sans'))
    return fig

# Automatically patch all plotly charts
px.defaults.template = None
px.defaults.width = None
px.defaults.height = None
px.defaults.labels = None
px.defaults.color_discrete_sequence = None

# Monkey patch st.plotly_chart to auto-style
_old_plotly_chart = st.plotly_chart
def styled_plotly_chart(fig, **kwargs):
    fig = update_transparent(fig)
    return _old_plotly_chart(fig, **kwargs)
st.plotly_chart = styled_plotly_chart




# --- Load All Resources ---
df = load_data()
rating_model, encoders, clf_pipeline = load_models()

# --- Sidebar ---
st.sidebar.title("▼ Explore the App")
page = st.sidebar.radio("Go to", [
    "Home",
    "Project Details",
    "Rating Predictor",
    "Visit Mode Predictor",
    "Recommender",
    "Dashboard Visualizations",
    "End page"
])
if page == "Home":
    st.markdown("""
        <div style='margin-top: -60px;'>
            <h1 style='text-align: center; font-size: 46px; color: black;'>
                TOURISM EXPERIENCE ANALYTICS✈︎
            </h1>
        </div>
        <p style='font-size: 24px; color: #555879; text-align: center; font-weight: bold;'>  
            "Tourism360: Explore✈︎Predict✈︎Recommend✈︎"
        </p>  

        <p style='font-size: 24px; color: #212121; text-align: left; font-weight: bold;'>      
            DOMAIN - TOURISM  
        </p>    
        <p>This application leverages historical and demographic data to:</p> 
        <ul>
            <li>Predict attraction ratings (regression)</li>
            <li>Classify user visit mode (classification)</li>
            <li>Recommend attractions (recommender systems)</li>
        </ul>
        <p style='font-size: 24px; color: #212121; text-align: left; font-weight: bold;'>           
            BUILT USING: 
        </p>
        <p>Pandas, SQL, Streamlit, Machine Learning (LightGBM, Random Forest), KNN, and more.</p>
    """, unsafe_allow_html=True)

elif page == "Project Details":
    st.title("Project Details")
    st.markdown("""
    ### Skills Takeaway
    - Data Cleaning and Preprocessing  
    - Exploratory Data Analysis (EDA)  
    - Data Visualization, SQL, Streamlit  
    - Machine Learning: Regression, Classification & Recommendation

    ### Business Use Cases
    - Personalized Recommendations
    - Tourism Analytics
    - Customer Segmentation
    - Increasing Customer Retention

    ### Objectives
    1. **Regression**: Predict ratings based on user and attraction data.
    2. **Classification**: Predict visit mode using user demographics.
    3. **Recommendation**: Suggest personalized attractions.
    """)

# --- Page 1: Rating Predictor ---
elif page == "Rating Predictor":
    st.header("Tourism Rating Predictor")
    attraction_map = dict(zip(df['Attraction'], df['AttractionId']))
    country_map = dict(zip(df['VisitorCountry'], df['VisitorCountryId']))
    continent_map = dict(zip(df['VisitorContinent'], df['VisitorContinentId']))
    city_map = dict(zip(df['VisitorCity'], df['VisitorCityId']))
    visit_modes = sorted(df['VisitMode'].unique())

    attraction = st.selectbox("Attraction", sorted(attraction_map))
    visit_mode = st.selectbox("Visit Mode", visit_modes)
    month = st.selectbox("Visit Month", list(range(1, 13)))
    year = st.number_input("Visit Year", 2000, 2030, step=1, value=2024)
    country = st.selectbox("Visitor Country", sorted(country_map))
    continent = st.selectbox("Visitor Continent", sorted(continent_map))
    city = st.selectbox("Visitor City", sorted(city_map))

    # --- PAGE CONTENTS ---
    
    

    if st.button("Predict Rating"):
        a_id = attraction_map[attraction]
        c_id = country_map[country]
        ct_id = continent_map[continent]
        city_id = city_map[city]
        season = get_season(month)
        attr_info = df[df['AttractionId'] == a_id].iloc[0]
        attr_city_id = attr_info['AttractionCityId']
        attr_type_id = attr_info['AttractionTypeId']
        region_id = attr_info['VisitorRegionId']
        user_data = df[df['VisitorCityId'] == city_id]
        total_visits = user_data.shape[0] or df['VisitorCityId'].value_counts().median()
        avg_rating = user_data['Rating'].mean() or df['Rating'].mean()
        diversity = user_data['AttractionId'].nunique() or df['AttractionId'].nunique()
        user_id = hash(f"{city_id}_{c_id}") % 10000
        input_dict = {
            'UserId': user_id,
            'VisitorCityId': city_id,
            'VisitorCountryId': c_id,
            'VisitorRegionId': region_id,
            'VisitorContinentId': ct_id,
            'AttractionTypeId': attr_type_id,
            'AttractionId': a_id,
            'AttractionCityId': attr_city_id,
            'VisitMode': visit_mode,
            'VisitMonth': month,
            'VisitYear': year,
            'User_TotalVisits': total_visits,
            'User_AvgRating': avg_rating,
            'User_AttractionDiversity': diversity,
            'User_VisitMode_MostUsed': visit_mode,
            'Is_SameCountry': int(c_id == attr_city_id),
            'Is_SameRegion': int(region_id == attr_city_id),
            'Is_SameContinent': int(ct_id == attr_city_id),
            'Season': season,
            'User_Attraction_Pair': f"{user_id}_{a_id}",
            'Country_VisitMode': f"{c_id}_{visit_mode}",
            'Region_VisitMonth': f"{region_id}_{month}",
            'Region_Season': f"{region_id}_{season}",
            'Continent_Season': f"{ct_id}_{season}",
            'VisitMode_Season': f"{visit_mode}_{season}",
            'User_PreferredMode_Season': f"{visit_mode}_{season}",
            'Attraction_City_Season': f"{attr_city_id}_{season}",
            'Attraction_Mode': f"{a_id}_{visit_mode}",
            'Visitor_Area': f"{region_id}_{attr_city_id}",
            'Attraction_AvgRating': df[df['AttractionId'] == a_id]['Rating'].mean()
        }
        input_df = pd.DataFrame([input_dict])
        for col in input_df.columns:
            if input_df[col].dtype == 'object' and col in encoders:
                val = input_df[col].iloc[0]
                input_df[col] = encoders[col].transform([val]) if val in encoders[col].classes_ else encoders[col].transform(['__unknown__'])
        pred_log = rating_model.predict(input_df)[0]
        st.success(f"Predicted Rating: {np.expm1(pred_log):.2f}")

# --- Page 2: Visit Mode Predictor ---
# In your Streamlit app (Visit Mode Predictor section)


elif page == "Visit Mode Predictor":
    st.header("Visit Mode Predictor")
    
    cm = dict(zip(df['VisitorContinent'], df['VisitorContinentId']))
    rm = dict(zip(df['VisitorRegion'], df['VisitorRegionId']))
    com = dict(zip(df['VisitorCountry'], df['VisitorCountryId']))
    citym = dict(zip(df['VisitorCity'], df['VisitorCityId']))
    am = dict(zip(df['Attraction'], df['AttractionId']))
    atm = dict(zip(df['AttractionType'], df['AttractionTypeId']))
    acm = dict(zip(df['AttractionCity'], df['AttractionCityId']))

    col1, col2 = st.columns(2)
    with col1:
        continent = st.selectbox("Continent", sorted(cm))
        region = st.selectbox("Region", sorted(rm))
        country = st.selectbox("Country", sorted(com))
        city = st.selectbox("City", sorted(citym))
    with col2:
        attraction = st.selectbox("Attraction", sorted(am))
        atype = st.selectbox("Attraction Type", sorted(atm))
        acity = st.selectbox("Attraction City", sorted(acm))
        year = st.number_input("Year", 2000, 2030, value=2024)
        month = st.selectbox("Month", list(range(1, 13)))

    col3, col4 = st.columns(2)
    with col3:
        total_visits = st.slider("Total Visits", 1, 100, 5)
    with col4:
        rating = st.slider("Rating", 1.0, 5.0, 4.5, step=0.1)

    if st.button("Predict Visit Mode"):
        input_df = pd.DataFrame([{
            'VisitorContinentId': cm[continent],
            'VisitorRegionId': rm[region],
            'VisitorCountryId': com[country],
            'VisitorCityId': citym[city],
            'AttractionId': am[attraction],
            'AttractionTypeId': atm[atype],
            'AttractionCityId': acm[acity],
            'VisitYear': year,
            'VisitMonth': month,
            'User_TotalVisits': total_visits,
            'Rating': rating,
            'User_AttractionDiversity': 3,
            'User_AvgRating': 4.0
        }])

        pred = clf_pipeline.predict(input_df)[0]
        probs = clf_pipeline.predict_proba(input_df)[0]
        labels = clf_pipeline.classes_

        st.success(f"Predicted Visit Mode: {pred}")

        # Show probabilities
        prob_df = pd.DataFrame({
            "VisitMode": labels,
            "Probability": probs
        }).sort_values(by="Probability", ascending=False)

        # st.write("Prediction Probabilities:")
        # st.dataframe(prob_df.style.format({"Probability": "{:.2%}"}))

# --- Page 3: Recommender ---
elif page == "Recommender":
    st.header("Attraction Recommender")
    knn, matrix, centered = train_recommender(df)
    users = sorted(df['UserId'].dropna().unique())
    uid = st.selectbox("Select User ID", users)
    topn = st.slider("Number of Recommendations", 1, 10, 5)
    if st.button("Get Recommendations"):
        recs, err = recommend_attractions(uid, knn, matrix, centered, df, top_n=topn)
        if err:
            st.warning(err)
        else:
            st.dataframe(recs)

# --- Page 4: Dashboard Visualizations ---


# --- Page 4: Dashboard Visualizations ---
elif page == "Dashboard Visualizations":
    st.title("Tourism Dashboard Visualizations")

    # Ensure your background is set earlier in your app using:
    # set_bg_from_local("/mnt/data/d3ae9af8-94a8-4b9f-a865-798f22bd0329.png")

    # Helper function to apply consistent transparent layout
    def update_transparent(fig):
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=14, color='black'),
            title_font=dict(size=20),
        )
        return fig

    # 1. Top 10 Most Visited Attractions
    # st.subheader("1. Top 10 Most Visited Attractions")
    top_attractions = df['Attraction'].value_counts().head(10).reset_index()
    top_attractions.columns = ['Attraction', 'Count']
    fig1 = px.bar(top_attractions, x='Attraction', y='Count', title="1. Top 10 Most Visited Attractions",
                  color_discrete_sequence=['#1f77b4'])
    fig1.update_xaxes(tickangle=45)
    st.plotly_chart(update_transparent(fig1), use_container_width=True)

    # 2️. Visit Mode Distribution by Continent
    # st.subheader("2. Visit Mode Distribution by Continent")
    visit_mode_dist = df.groupby(['VisitorContinent', 'VisitMode']).size().reset_index(name='Count')
    fig2 = px.bar(visit_mode_dist, x='VisitorContinent', y='Count', color='VisitMode',
                  title="2. Visit Mode Distribution by Continent", barmode='stack')
    st.plotly_chart(update_transparent(fig2), use_container_width=True)

    # 3️⃣ Average Rating by Attraction Type
    # st.subheader("3️⃣ Average Rating by Attraction Type")
    avg_rating_type = df.groupby('AttractionType')['Rating'].mean().sort_values(ascending=False).reset_index()
    fig3 = px.bar(avg_rating_type, x='AttractionType', y='Rating',
                  title="3. Average Rating by Attraction Type", color_discrete_sequence=['#ff7f0e'])
    fig3.update_xaxes(tickangle=45)
    st.plotly_chart(update_transparent(fig3), use_container_width=True)

    # 4️⃣ Monthly Visit Trends
    # st.subheader("4️⃣ Monthly Visit Trends")
    monthly_visits = df['VisitMonth'].value_counts().sort_index().reset_index()
    monthly_visits.columns = ['Month', 'VisitCount']
    fig4 = px.line(monthly_visits, x='Month', y='VisitCount', markers=True,
                   title="4. Monthly Visit Trends", line_shape="linear")
    st.plotly_chart(update_transparent(fig4), use_container_width=True)

    # 5️⃣ Additional Visualizations
    # st.subheader("5️⃣ Additional Visualizations")

    # A. Average Rating by Country (Top 15)
    # st.markdown("📈 **5. Average Rating by Country**")
    country_ratings = df.groupby('VisitorCountry')['Rating'].mean().sort_values(ascending=False).head(15).reset_index()
    fig5 = px.line(country_ratings, x='VisitorCountry', y='Rating', markers=True,
                   title="5. Average Rating by Country (Top 15)")
    fig5.update_xaxes(tickangle=45)
    st.plotly_chart(update_transparent(fig5), use_container_width=True)

    # B. Visit Mode Distribution by Top 5 Countries
    # st.markdown("📊 **6. Visit Mode Distribution by Top 5 Countries**")
    top_countries = df['VisitorCountry'].value_counts().head(5).index.tolist()
    subset = df[df['VisitorCountry'].isin(top_countries)]
    mode_by_country = subset.groupby(['VisitorCountry', 'VisitMode']).size().reset_index(name='Count')
    fig6 = px.bar(mode_by_country, x='VisitorCountry', y='Count', color='VisitMode',
                  title="6. Visit Mode Distribution by Top 5 Countries", barmode='stack')
    st.plotly_chart(update_transparent(fig6), use_container_width=True)


elif page == "End page":
    st.markdown("""
    ### Final Summary
    - Models implemented: Regression, Classification, Recommender  
    - Streamlit Interface for interaction and visualization  
    - SQL integration for real-time data fetching

    ### Evaluation Metrics
    - Regression: R², RMSE
    - Classification: Accuracy, F1-score
    - Recommender: MAP      
    --- 
     </p>
            <p style='font-size: 16px; color: #212121; text-align: Left; font-weight: bold;'>
      You've reached the end of your journey through the "Tourism experience analytics". </p> 
      We hope this platform helped you discover insights into tourist behavior and attraction engagement. 
                 
     </p> 
            <p style='font-size: 28px; color: #555879; text-align: center; font-weight: bold;'>  
             Thank you for visiting!   </p>
            <p style='font-size: 30px; color: #555879; text-align: center; font-weight: bold;'>    
             Explore More ✈︎ Discover More ✈︎    
        
    """, unsafe_allow_html=True) 
    


    # ---
    # You've reached the end of your journey through the **Tourism Intelligence App** 🌍.  
    # We hope this platform helped you discover insights into tourist behavior and attraction engagement.  

    # ### 🛠️ Built With ❤️ Using:
    # 1. **Streamlit** – For an interactive user experience  
    # 2. **Pandas & MySQL** – For data management and storage  
    # 3. **Scikit-learn, LightGBM & KNN** – For powerful machine learning models  

    # ✨ *Explore More. Discover More. Predict Smarter.*

    # Thank you for visiting!
    # """)