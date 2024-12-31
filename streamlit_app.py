import streamlit as st
import pandas as pd
import google.generativeai as genai
from openai import OpenAI
import requests
import json
from typing import Dict, List, Optional
from PIL import Image

# API Keys
OPENAI_API_KEY = "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"
GEMINI_API_KEY = "AIzaSyCcMZPrzP5me7Rl4pmAc1Nn5vUDSan5Q6E"
PERPLEXITY_API_KEY = "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"
GROQ_API_KEY = "groq-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"

class GarageDataManager:
    def __init__(self):
        self.data = None

    def load_csv(self, file) -> Optional[pd.DataFrame]:
        try:
            self.data = pd.read_csv(file)
            self._clean_data()
            return self.data
        except Exception as e:
            st.error(f"Error loading CSV: {str(e)}")
            return None

    def _clean_data(self):
        if self.data is None:
            return

        # Clean string columns
        string_columns = ['GarageName', 'Location', 'City', 'Postcode', 
                         'Phone', 'Email', 'Website']
        for col in string_columns:
            if col in self.data.columns:
                self.data[col] = self.data[col].astype(str).str.strip()

        # Standardize postcode
        if 'Postcode' in self.data.columns:
            self.data['Postcode'] = self.data['Postcode'].str.upper()

    def search_garages(self, query: str, search_by: str) -> pd.DataFrame:
        if self.data is None:
            return pd.DataFrame()

        query = str(query).lower()
        if search_by == "City":
            return self.data[self.data['City'].str.lower().str.contains(query)]
        elif search_by == "Postcode":
            return self.data[self.data['Postcode'].str.lower().str.contains(query)]
        elif search_by == "Name":
            return self.data[self.data['GarageName'].str.lower().str.contains(query)]
        return pd.DataFrame()

class AIServices:
    def __init__(self):
        # Initialize OpenAI
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Initialize Gemini
        genai.configure(api_key=GEMINI_API_KEY)
        self.gemini = genai.GenerativeModel('gemini-pro')

    def generate_image(self, prompt: str) -> str:
        try:
            response = self.openai_client.images.generate(
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

    def get_expert_advice(self, query: str) -> str:
        try:
            response = self.openai_client.chat.completions.create(
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

    def research_with_perplexity(self, query: str) -> str:
        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama-3.1-sonar-small-128k-online",
            "messages": [
                {"role": "system", "content": "Research automotive topics."},
                {"role": "user", "content": query}
            ]
        }

        try:
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

    def analyze_with_gemini(self, query: str) -> str:
        try:
            response = self.gemini.generate_content(query)
            return response.text
        except Exception as e:
            st.error(f"Gemini analysis error: {str(e)}")
            return None

def main():
    st.set_page_config(page_title="Auto Service Assistant", layout="wide")
    
    # Initialize services
    if 'ai_services' not in st.session_state:
        st.session_state.ai_services = AIServices()
    if 'garage_manager' not in st.session_state:
        st.session_state.garage_manager = GarageDataManager()

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
            data = st.session_state.garage_manager.load_csv(uploaded_file)
            if data is not None:
                st.success(f"Loaded {len(data)} garages")

    # Main interface
    tabs = st.tabs(["Search", "Research", "Design", "Advisor"])

    with tabs[0]:
        st.header("🔍 Garage Search")
        if hasattr(st.session_state.garage_manager, 'data'):
            col1, col2 = st.columns([2, 1])
            with col1:
                search_query = st.text_input("Search Garages")
            with col2:
                search_by = st.selectbox("Search By", ["City", "Postcode", "Name"])

            if search_query:
                results = st.session_state.garage_manager.search_garages(search_query, search_by)
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

    with tabs[1]:
        st.header("🔬 Part Research")
        research_query = st.text_area("What would you like to research?")
        if st.button("Research"):
            with st.spinner("Researching..."):
                result = st.session_state.ai_services.research_with_perplexity(
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
                image_url = st.session_state.ai_services.generate_image(prompt)
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
                advice = st.session_state.ai_services.get_expert_advice(
                    f"Provide expert advice for: {issue} "
                    f"Vehicle: {year} {make} {model}"
                )
                if advice:
                    st.write(advice)
                    
                    # Show relevant garages if data is loaded
                    if hasattr(st.session_state.garage_manager, 'data'):
                        st.subheader("Nearby Garages")
                        # Add garage recommendations based on the issue

if __name__ == "__main__":
    main()
