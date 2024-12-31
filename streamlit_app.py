import streamlit as st
import pandas as pd
import google.generativeai as genai
from openai import OpenAI
import requests
from typing import Dict, List, Optional

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.garage_data = None
    st.session_state.openai_client = None
    st.session_state.gemini_model = None

class GarageDataManager:
    @staticmethod
    def load_csv(file) -> Optional[pd.DataFrame]:
        try:
            data = pd.read_csv(file)
            # Clean data
            string_columns = ['GarageName', 'Location', 'City', 'Postcode', 
                            'Phone', 'Email', 'Website']
            for col in string_columns:
                if col in data.columns:
                    data[col] = data[col].astype(str).str.strip()
            
            if 'Postcode' in data.columns:
                data['Postcode'] = data['Postcode'].str.upper()
            
            return data
        except Exception as e:
            st.error(f"Error loading CSV: {str(e)}")
            return None

    @staticmethod
    def search_garages(data: pd.DataFrame, query: str, search_by: str) -> pd.DataFrame:
        if data is None:
            return pd.DataFrame()

        query = str(query).lower()
        if search_by == "City":
            return data[data['City'].str.lower().str.contains(query)]
        elif search_by == "Postcode":
            return data[data['Postcode'].str.lower().str.contains(query)]
        elif search_by == "Name":
            return data[data['GarageName'].str.lower().str.contains(query)]
        return pd.DataFrame()

def initialize_ai_services():
    """Initialize AI services and store in session state"""
    if not st.session_state.initialized:
        try:
            # Initialize OpenAI
            st.session_state.openai_client = OpenAI(
                api_key="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"
            )
        except Exception as e:
            st.warning("OpenAI service initialization failed")
            st.session_state.openai_client = None

        try:
            # Initialize Gemini
            genai.configure(api_key="AIzaSyCcMZPrzP5me7Rl4pmAc1Nn5vUDSan5Q6E")
            st.session_state.gemini_model = genai.GenerativeModel('gemini-pro')
        except Exception as e:
            st.warning("Gemini service initialization failed")
            st.session_state.gemini_model = None

        st.session_state.initialized = True

def generate_image(prompt: str) -> Optional[str]:
    """Generate image using DALL-E"""
    if st.session_state.openai_client is None:
        st.warning("Image generation service not available")
        return None

    try:
        response = st.session_state.openai_client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1
        )
        return response.data[0].url
    except Exception as e:
        st.error(f"Image generation error: {str(e)}")
        return None

def get_expert_advice(query: str) -> Optional[str]:
    """Get expert advice using GPT-4"""
    if st.session_state.openai_client is None:
        st.warning("Expert advice service not available")
        return None

    try:
        response = st.session_state.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an automotive expert."},
                {"role": "user", "content": query}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Expert advice error: {str(e)}")
        return None

def research_with_perplexity(query: str) -> Optional[str]:
    """Research using Perplexity API"""
    try:
        headers = {
            "Authorization": f"Bearer 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",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama-3.1-sonar-small-128k-online",
            "messages": [
                {"role": "system", "content": "Research automotive topics."},
                {"role": "user", "content": query}
            ]
        }

        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=payload
        )
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        st.error(f"Research error: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="Auto Service Assistant", layout="wide")
    
    # Initialize AI services
    initialize_ai_services()

    st.title("🚗 Automotive Service Assistant")

    # Sidebar
    with st.sidebar:
        st.header("Vehicle Information")
        make = st.text_input("Make")
        model = st.text_input("Model")
        year = st.number_input("Year", min_value=1900, max_value=2024)

        st.header("Data Upload")
        uploaded_file = st.file_uploader("Upload Garage CSV", type=['csv'])
        if uploaded_file is not None:
            st.session_state.garage_data = GarageDataManager.load_csv(uploaded_file)
            if st.session_state.garage_data is not None:
                st.success(f"Loaded {len(st.session_state.garage_data)} garages")

    # Main interface
    tabs = st.tabs(["Search", "Research", "Design", "Advisor"])

    with tabs[0]:
        st.header("🔍 Garage Search")
        if st.session_state.garage_data is not None:
            col1, col2 = st.columns([2, 1])
            with col1:
                search_query = st.text_input("Search Garages")
            with col2:
                search_by = st.selectbox("Search By", ["City", "Postcode", "Name"])

            if search_query:
                results = GarageDataManager.search_garages(
                    st.session_state.garage_data, 
                    search_query, 
                    search_by
                )
                if not results.empty:
                    for _, garage in results.iterrows():
                        with st.expander(f"{garage['GarageName']} - {garage['City']}"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**Location:**", garage['Location'])
                                st.write("**City:**", garage['City'])
                                st.write("**Postcode:**", garage['Postcode'])
                            with col2:
                                st.write("**Phone:**", garage['Phone'])
                                st.write("**Email:**", garage['Email'])
                                if pd.notna(garage['Website']):
                                    st.write("**Website:**", garage['Website'])
        else:
            st.info("Please upload garage data CSV to enable search.")

    with tabs[1]:
        st.header("🔬 Part Research")
        research_query = st.text_area("What would you like to research?")
        if st.button("Research"):
            with st.spinner("Researching..."):
                result = research_with_perplexity(
                    f"Research automotive topic: {research_query} for {year} {make} {model}"
                )
                if result:
                    st.write(result)

    with tabs[2]:
        st.header("🎨 Design Visualization")
        design_type = st.selectbox(
            "What would you like to visualize?",
            ["Custom Paint", "Body Modifications", "Interior Design", "Parts"]
        )
        design_desc = st.text_area("Describe your design")
        
        if st.button("Generate Design"):
            with st.spinner("Creating design..."):
                prompt = f"Professional automotive {design_type.lower()} design: {design_desc}"
                image_url = generate_image(prompt)
                if image_url:
                    st.image(image_url)
                    st.download_button(
                        "Download Design",
                        image_url,
                        "design.png",
                        "image/png"
                    )

    with tabs[3]:
        st.header("💡 Expert Advisor")
        issue = st.text_area("Describe your car issue or service need")
        
        if st.button("Get Advice"):
            with st.spinner("Analyzing..."):
                advice = get_expert_advice(
                    f"Provide expert advice for: {issue} "
                    f"Vehicle: {year} {make} {model}"
                )
                if advice:
                    st.write(advice)

if __name__ == "__main__":
    main()
