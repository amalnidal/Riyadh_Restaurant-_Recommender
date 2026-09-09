
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Page configuration
st.set_page_config(
    page_title="Riyadh Restaurant Recommender",
    page_icon="🍽️",
    layout="wide"
)

# Title
st.title("🍽️ AI-Powered Restaurant Recommender for Riyadh")
st.markdown("Find your perfect dining experience based on your preferences!")

# Load the model
@st.cache_resource
def load_model():
    try:
        pipeline = joblib.load("riyadh_recommender_pipeline.pkl")
        return pipeline
    except:
        st.error("Pipeline file not found. Please ensure 'riyadh_recommender_pipeline.pkl' is in the same directory.")
        return None

pipeline = load_model()

if pipeline is not None:
    # Extract components
    model = pipeline["model"]
    vectorizer = pipeline["vectorizer"]
    restaurants_df = pipeline["restaurants_df"]
    features = pipeline["features"]
    
    # Get unique values for dropdowns
    budget_options = sorted(restaurants_df['price'].unique().tolist())
    budget_options = [b for b in budget_options if b != 'Unknown']
    
    # Sidebar - User Inputs
    st.sidebar.header("Your Preferences")
    
    # Cuisine input
    user_cuisine = st.sidebar.text_input(
        "What cuisine are you craving?",
        value="coffee shop",
        help="e.g., pizza, burger, coffee, indian, etc."
    )
    
    # Budget input
    user_budget = st.sidebar.selectbox(
        "What's your budget?",
        options=budget_options,
        index=0 if 'Cheap' in budget_options else 0
    )
    
    # Rating input
    min_rating = st.sidebar.slider(
        "Minimum Rating (out of 10)",
        min_value=1.0,
        max_value=10.0,
        value=7.0,
        step=0.5
    )
    
    # Number of recommendations
    num_recommendations = st.sidebar.slider(
        "Number of recommendations",
        min_value=5,
        max_value=20,
        value=10,
        step=5
    )
    
    # Generate recommendations button
    if st.sidebar.button("Find Restaurants", use_container_width=True):
        
        with st.spinner("Finding the best restaurants for you..."):
            # Convert user cuisine to TF-IDF
            user_cuisine_vec = vectorizer.transform([user_cuisine])
            
            # Calculate cosine similarity
            similarities = cosine_similarity(user_cuisine_vec, vectorizer.transform(restaurants_df['categories'])).flatten()
            
            # Create features
            temp_df = restaurants_df.copy()
            temp_df['cuisine_similarity'] = similarities
            temp_df['budget_match'] = (temp_df['price'] == user_budget).astype(int)
            temp_df['rating_match'] = (
                temp_df['rating'].notna() & 
                (temp_df['rating'] >= min_rating)
            ).astype(int)
            
            # Predict probabilities
            X = temp_df[features]
            temp_df['like_probability'] = model.predict_proba(X)[:, 1]
            
            # Filter by budget and rating
            filtered = temp_df[
                (temp_df['price'] == user_budget) &
                (temp_df['rating'].notna()) &
                (temp_df['rating'] >= min_rating)
            ]
            
            # Sort by probability
            filtered = filtered.sort_values('like_probability', ascending=False)
            
            # Get top N
            recommendations = filtered.head(num_recommendations)
        
        if len(recommendations) == 0:
            st.warning("No restaurants found matching your criteria. Try adjusting your preferences.")
        else:
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Found Restaurants", len(recommendations))
            with col2:
                avg_rating = recommendations['rating'].mean()
                st.metric("Average Rating", f"{avg_rating:.1f}/10")
            with col3:
                avg_match = recommendations['like_probability'].mean()
                st.metric("Average Match", f"{avg_match:.1%}")
            with col4:
                top_match = recommendations['like_probability'].max()
                st.metric("Best Match", f"{top_match:.1%}")
            
            # Display recommendations
            st.subheader("Top Restaurant Recommendations")
            
            for idx, (_, row) in enumerate(recommendations.iterrows()):
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"### {idx+1}. {row['name']}")
                        st.write(f"**Categories:** {row['categories']}")
                        st.write(f"**Rating:** {row['rating']:.1f}/10")
                        st.write(f"**Match Score:** {row['like_probability']:.1%}")
                    with col2:
                        price_icons = {'Cheap': '$', 'Moderate': '$$', 'Expensive': '$$$', 'Very Expensive': '$$$$'}
                        st.metric("Price", f"{row['price']} {price_icons.get(row['price'], '')}")
                        if row['like_probability'] > 0.8:
                            st.success("Excellent Match")
                        elif row['like_probability'] > 0.6:
                            st.info("Good Match")
                        else:
                            st.warning("Decent Match")
                    st.divider()
                    
else:
    st.error("Failed to load the recommendation pipeline. Please check your files.")

st.markdown("---")
st.markdown("Built with using Streamlit and scikit-learn")
