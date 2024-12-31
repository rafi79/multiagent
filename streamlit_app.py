import streamlit as st
import pandas as pd
import google.generativeai as genai
from openai import OpenAI
import requests
import json
import io
import base64
from datetime import datetime
from typing import Dict, List, Optional, Union
from PIL import Image

# Set page config
st.set_page_config(page_title="Automotive Multi-Agent System", layout="wide")

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

class GarageDataManager:
    def __init__(self):
        self.data = None
        
    def load_csv(self, file) -> Optional[pd.DataFrame]:
        try:
            self.data = pd.read_csv(file, names=[
                'Serial', 'GarageName', 'Location', 'City', 
                'Postcode', 'Phone', 'Email', 'Website'
            ], skiprows=1)
            self._clean_data()
            return self.data
        except Exception as e:
            st.error(f"Error loading CSV: {str(e)}")
            return None
            
    def _clean_data(self):
        if self.data is None:
            return
            
        string_columns = ['GarageName', 'Location', 'City', 'Postcode', 
                         'Phone', 'Email', 'Website']
        for col in string_columns:
            if col in self.data.columns:
                self.data[col] = self.data[col].astype(str).str.strip()
                
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

class MultiAgentSystem:
    def __init__(self):
        # Initialize Gemini
        try:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            self.gemini = genai.GenerativeModel('gemini-pro')
        except Exception as e:
            st.warning("Gemini API not configured")
            self.gemini = None

        # Initialize OpenAI
        try:
            self.openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        except Exception as e:
            st.warning("OpenAI API not configured")
            self.openai_client = None

        # Perplexity Setup
        self.perplexity_key = st.secrets.get("PERPLEXITY_API_KEY")
        
    def analyze_with_gemini(self, query: str) -> str:
        if not self.gemini:
            return "Gemini API not configured"
        try:
            response = self.gemini.generate_content(query)
            return response.text
        except Exception as e:
            st.error(f"Gemini API error: {str(e)}")
            return None
            
    def generate_image_with_dalle(self, prompt: str) -> str:
        if not self.openai_client:
            return "OpenAI API not configured"
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
            st.error(f"DALL-E API error: {str(e)}")
            return None
            
    def reason_with_o1(self, query: str) -> str:
        if not self.openai_client:
            return "OpenAI API not configured"
        try:
            response = self.openai_client.chat.completions.create(
                model="o1-mini",
                messages=[
                    {"role": "system", "content": "You are an automotive expert."},
                    {"role": "user", "content": query}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"O1 API error: {str(e)}")
            return None
            
    def research_with_perplexity(self, query: str) -> str:
        if not self.perplexity_key:
            return "Perplexity API not configured"
            
        headers = {
            "Authorization": f"Bearer {self.perplexity_key}",
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
            st.error(f"Perplexity API error: {str(e)}")
            return None

def initialize_system():
    if not st.session_state.initialized:
        try:
            st.session_state.garage_manager = GarageDataManager()
            st.session_state.agent_system = MultiAgentSystem()
            st.session_state.initialized = True
        except Exception as e:
            st.error(f"Error initializing system: {str(e)}")

def main():
    st.title("🚗 Automotive Multi-Agent System")
    
    # Initialize system
    initialize_system()
    
    # Sidebar - Vehicle Info
    with st.sidebar:
        st.header("Vehicle Information")
        make = st.text_input("Make")
        model = st.text_input("Model")
        year = st.number_input("Year", min_value=1900, max_value=2024)
        
        st.header("Data Management")
        uploaded_file = st.file_uploader("Upload Garage CSV", type=['csv'])
        if uploaded_file is not None:
            data = st.session_state.garage_manager.load_csv(uploaded_file)
            if data is not None:
                st.success(f"Loaded {len(data)} records")
    
    # Main tabs
    tabs = st.tabs([
        "Garage Search",
        "Part Research",
        "Visual Design",
        "Service Advisor"
    ])
    
    with tabs[0]:
        st.header("🔍 Find Garages")
        if hasattr(st.session_state.garage_manager, 'data') and st.session_state.garage_manager.data is not None:
            col1, col2 = st.columns([2, 1])
            with col1:
                search_query = st.text_input("Search")
            with col2:
                search_by = st.selectbox("Search By", ["City", "Postcode", "Name"])
            
            if search_query:
                results = st.session_state.garage_manager.search_garages(search_query, search_by)
                if not results.empty:
                    st.write(f"Found {len(results)} matches:")
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
        research_query = st.text_area("What part or service do you need information about?")
        
        if st.button("Research"):
            with st.spinner("Researching..."):
                research_result = st.session_state.agent_system.research_with_perplexity(
                    f"Research automotive part/service: {research_query} for {year} {make} {model}"
                )
                if research_result:
                    st.write(research_result)
    
    with tabs[2]:
        st.header("🎨 Visual Design")
        design_type = st.selectbox(
            "What would you like to visualize?",
            ["Custom Paint", "Body Modifications", "Interior Design", "Parts"]
        )
        
        design_desc = st.text_area("Describe your design")
        
        if st.button("Generate Design"):
            with st.spinner("Creating design..."):
                image_url = st.session_state.agent_system.generate_image_with_dalle(
                    f"Professional automotive {design_type.lower()} design: {design_desc}"
                )
                if image_url:
                    st.image(image_url)
                    st.download_button(
                        "Download Design",
                        image_url,
                        "design.png",
                        "image/png"
                    )
    
    with tabs[3]:
        st.header("💡 Service Advisor")
        issue = st.text_area("Describe your car issue or service need")
        
        if st.button("Get Advice"):
            with st.spinner("Analyzing..."):
                advice = st.session_state.agent_system.reason_with_o1(
                    f"Provide expert advice for this automotive issue: {issue} "
                    f"Vehicle: {year} {make} {model}"
                )
                if advice:
                    st.write(advice)

if __name__ == "__main__":
    main()
