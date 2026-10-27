import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Athlete Brand Matchmaker",
    page_icon="🏆",
    layout="wide"
)

st.title("🏆 Athlete–Brand Deal Matchmaker")
st.write(
    "Discover potential sponsorship matches using athlete interests, "
    "audience alignment, brand values, and estimated reach."
)

@st.cache_data
def load_data():
    athletes = pd.read_csv("athletes - Sheet1.csv")
    brands = pd.read_csv("brands - Sheet1.csv")
    athletes.columns = athletes.columns.str.strip().str.lower()
    brands.columns = brands.columns.str.strip().str.lower()
    return athletes, brands

try:
    athletes, brands = load_data()
except Exception as e:
    st.error("Could not load the athlete and brand CSV files.")
    st.write("Make sure both CSV files are included in the project.")
    st.stop()

required_athlete_cols = {
    "athlete", "interests", "audience", "values", "followers"
}
required_brand_cols = {
    "brand", "categories", "audience", "values", "ideal_reach"
}

if not required_athlete_cols.issubset(athletes.columns):
    st.error("The athlete CSV is missing one or more required columns.")
    st.stop()

if not required_brand_cols.issubset(brands.columns):
    st.error("The brand CSV is missing one or more required columns.")
    st.stop()

def text_similarity(text1, text2):
    text1 = str(text1)
    text2 = str(text2)

    if not text1.strip() or not text2.strip():
        return 0.0

    try:
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform([text1, text2])
        return float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0]) * 100
    except ValueError:
        return 0.0

def reach_score(followers, ideal_reach):
    try:
        followers = float(followers)
        ideal_reach = float(ideal_reach)
        if followers <= 0 or ideal_reach <= 0:
            return 0.0
        ratio = followers / ideal_reach
        return max(0.0, 100.0 - abs(ratio - 1.0) * 50.0)
    except (ValueError, TypeError):
        return 0.0

def calculate_match(athlete, brand):
    product = text_similarity(
        athlete["interests"], brand["categories"]
    )
    audience = text_similarity(
        athlete["audience"], brand["audience"]
    )
    values = text_similarity(
        athlete["values"], brand["values"]
    )
    reach = reach_score(
        athlete["followers"], brand["ideal_reach"]
    )

    total = (
        0.45 * product
        + 0.25 * audience
        + 0.20 * values
        + 0.10 * reach
    )

    return {
        "Athlete": athlete["athlete"],
        "Brand": brand["brand"],
        "Match Score": round(total, 1),
        "Product Fit": round(product, 1),
        "Audience Fit": round(audience, 1),
        "Values Fit": round(values, 1),
        "Reach Fit": round(reach, 1),
    }

athlete_names = sorted(athletes["athlete"].dropna().unique())

selected_athlete = st.selectbox(
    "Choose an athlete",
    athlete_names
)

athlete_row = athletes[
    athletes["athlete"] == selected_athlete
].iloc[0]

results = [
    calculate_match(athlete_row, brand)
    for _, brand in brands.iterrows()
]

results_df = pd.DataFrame(results).sort_values(
    "Match Score", ascending=False
).reset_index(drop=True)

st.subheader(f"Best brand matches for {selected_athlete}")
st.caption(
    "Scores are prototype estimates, not probabilities of securing a deal. "
    "Results depend on the quality of the entered data and scoring assumptions."
)

st.dataframe(
    results_df,
    use_container_width=True,
    hide_index=True
)

st.subheader("Explore a match")

selected_brand = st.selectbox(
    "Choose a brand to inspect",
    results_df["Brand"].tolist()
)

match = results_df[results_df["Brand"] == selected_brand].iloc[0]

col1, col2, col3 = st.columns(3)
col1.metric("Overall match", f'{match["Match Score"]}/100')
col2.metric("Product fit", f'{match["Product Fit"]}/100')
col3.metric("Audience fit", f'{match["Audience Fit"]}/100')

col4, col5 = st.columns(2)
col4.metric("Values fit", f'{match["Values Fit"]}/100')
col5.metric("Reach fit", f'{match["Reach Fit"]}/100')

st.info(
    "This is an early-stage matching prototype. Similarity scores and "
    "reach assumptions should be validated with reliable, sourced data "
    "before using the rankings for real sponsorship decisions."
)
