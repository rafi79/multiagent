import streamlit as st
import pandas as pd
import google.generativeai as genai
import requests
from typing import Dict, List, Optional
import re
import os
from dotenv import load_dotenv
import logging
from datetime import datetime
import time
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('garage_assistant.log'),
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv()

class AIServices:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.mistral_client = None  # Initialize Mistral client here
        
    def generate_car_image(self, prompt: str) -> Dict:
        """Generate car customization image using DALL-E 3"""
        try:
            response = self.openai_client.images.generate(
                model="dall-e-3",
                prompt=f"Automotive design visualization: {prompt}",
                size="1024x1024",
                quality="standard",
                n=1
            )
            return {
                "success": True,
                "image_url": response.data[0].url,
                "error": None
            }
        except Exception as e:
            logging.error(f"Image generation error: {str(e)}")
            return {
                "success": False,
                "image_url": None,
                "error": str(e)
            }
    
    def analyze_customization(self, description: str) -> Dict:
        """Analyze car customization feasibility using Perplexity for web research"""
        try:
            # Perplexity web search for customization research
            headers = {
                "Authorization": f"Bearer {os.getenv('PERPLEXITY_API_KEY')}",
                "Content-Type": "application/json"
            }
            
            search_query = f"""Research the following car modification request and provide:
            1. Technical feasibility
            2. Estimated cost range
            3. Common challenges
            4. Required expertise
            5. Examples of similar modifications
            
            Modification request: {description}"""
            
            payload = {
                "model": "llama-3.1-sonar-small-128k-online",
                "messages": [
                    {"role": "user", "content": search_query}
                ],
                "context": {
                    "web_search": True,
                    "system_prompt": "You are an automotive customization expert. Use real-world examples and current market information in your analysis."
                }
            }

            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code != 200:
                raise Exception(f"Perplexity API error: {response.status_code}")
            
            analysis = response.json()['choices'][0]['message']['content']
            
            # Additional market research query
            market_payload = {
                "model": "llama-3.1-sonar-small-128k-online",
                "messages": [
                    {"role": "user", "content": f"Find recent examples and current market trends for this type of car modification: {description}"}
                ],
                "context": {
                    "web_search": True
                }
            }
            
            market_response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers=headers,
                json=market_payload
            )
            
            if market_response.status_code == 200:
                market_analysis = market_response.json()['choices'][0]['message']['content']
                combined_analysis = f"Technical Analysis:\n{analysis}\n\nMarket Research:\n{market_analysis}"
            else:
                combined_analysis = analysis
            
            return {
                "success": True,
                "analysis": combined_analysis,
                "error": None
            }
        except Exception as e:
            logging.error(f"Customization analysis error: {str(e)}")
            return {
                "success": False,
                "analysis": None,
                "error": str(e)
            }

class GarageAssistant:
    def __init__(self):
        # Initialize services
        self.ai_services = AIServices()
        self.rate_limiters = {
            'gemini': RateLimiter(calls=60, period=60),
            'openai': RateLimiter(calls=50, period=60),
            'perplexity': RateLimiter(calls=30, period=60)
        }
        
        # Initialize Gemini
        try:
            genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
            self.gemini = genai.GenerativeModel('gemini-pro')
        except Exception as e:
            logging.error(f"Failed to initialize Gemini API: {str(e)}")
            st.warning("Gemini service not available")
            self.gemini = None

    def process_customization_request(self, request: str, garage_data: pd.DataFrame) -> Dict:
        """Process car customization request with image generation"""
        if not self.rate_limiters['openai'].can_call():
            return {
                "message": "Rate limit exceeded for image generation. Please try again later.",
                "image_url": None,
                "analysis": None,
                "garages": [],
                "type": "error"
            }

        # Generate visualization
        image_result = self.ai_services.generate_car_image(request)
        if not image_result['success']:
            return {
                "message": f"Error generating visualization: {image_result['error']}",
                "image_url": None,
                "analysis": None,
                "garages": [],
                "type": "error"
            }

        # Analyze feasibility
        analysis_result = self.ai_services.analyze_customization(request)
        if not analysis_result['success']:
            return {
                "message": f"Error analyzing request: {analysis_result['error']}",
                "image_url": image_result['image_url'],
                "analysis": None,
                "garages": [],
                "type": "error"
            }

        # Find relevant garages
        relevant_garages = self._find_suitable_garages(request, garage_data)

        return {
            "message": "Here's a visualization of your custom design request:",
            "image_url": image_result['image_url'],
            "analysis": analysis_result['analysis'],
            "garages": relevant_garages.to_dict('records'),
            "type": "customization"
        }

    def _find_suitable_garages(self, request: str, garage_data: pd.DataFrame) -> pd.DataFrame:
        """Find garages suitable for the customization request"""
        # Extract relevant keywords from request
        keywords = ['paint', 'custom', 'body', 'modification', 'tune', 'performance']
        relevant_terms = [term for term in keywords if term in request.lower()]
        
        if not relevant_terms:
            return garage_data.head(0)  # Return empty if no relevant terms
            
        # Filter garages based on services
        mask = garage_data['GarageName'].str.lower().str.contains('|'.join(relevant_terms), regex=True)
        return garage_data[mask]

def main():
    st.set_page_config(
        page_title="Auto Service Assistant",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🚗 Automotive Service & Customization Assistant")

    # Initialize session state
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
        st.session_state.assistant = GarageAssistant()
        st.session_state.garage_data = None
        st.session_state.chat_history = []
        st.session_state.current_query = ""

    # Sidebar
    with st.sidebar:
        st.header("Data Upload")
        uploaded_file = st.file_uploader(
            "Upload Garage Database (CSV)",
            type=['csv'],
            help="Upload a CSV file containing garage information"
        )
        
        if uploaded_file is not None:
            try:
                st.session_state.garage_data = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(st.session_state.garage_data)} garages")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")

        st.header("Services")
        service_type = st.radio(
            "Select Service Type:",
            ["Regular Service", "Custom Design & Modification"]
        )

    # Main interface
    if service_type == "Custom Design & Modification":
        st.write("### 🎨 Custom Car Design Visualization")
        custom_request = st.text_area(
            "Describe your custom design idea:",
            placeholder="e.g., Custom pearl white paint job with blue racing stripes and carbon fiber hood"
        )
        
        if st.button("Generate Design Preview"):
            if st.session_state.garage_data is None:
                st.error("Please upload a garage database first!")
                return
                
            with st.spinner("🎨 Generating your custom design..."):
                result = st.session_state.assistant.process_customization_request(
                    custom_request,
                    st.session_state.garage_data
                )
                
                if result['image_url']:
                    st.image(result['image_url'], caption="Your Custom Design Preview")
                
                if result['analysis']:
                    st.write("### 📊 Feasibility Analysis")
                    st.write(result['analysis'])
                
                if result['garages']:
                    st.write("### 🏢 Recommended Garages")
                    for garage in result['garages']:
                        with st.expander(f"🔧 {garage['GarageName']} - {garage['City']}"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("📍 **Location:**", garage['Location'])
                                st.write("🏙️ **City:**", garage['City'])
                                st.write("📮 **Postcode:**", garage['Postcode'])
                            with col2:
                                st.write("📞 **Phone:**", garage['Phone'])
                                st.write("📧 **Email:**", garage['Email'])
                                if 'Website' in garage and pd.notna(garage['Website']):
                                    st.write("🌐 **Website:**", garage['Website'])
    else:
        # Regular service interface
        st.write("### Ask me about:")
        st.write("- 🔍 Finding specific services in your area")
        st.write("- 🎨 Paint and bodywork services")
        st.write("- ⭐ Garage recommendations")

        query = st.text_input(
            "Your question:",
            value=st.session_state.current_query,
            placeholder="e.g., Find garages in London specializing in paint work"
        )
        
        if query and st.button("🔍 Search"):
            if st.session_state.garage_data is None:
                st.error("Please upload a garage database first!")
                return

            with st.spinner("Processing your request..."):
                response = st.session_state.assistant.process_query(
                    query,
                    st.session_state.garage_data
                )
                st.write(f"🤖 **Assistant:** {response['message']}")
                
                if response['garages']:
                    st.write("### 🏢 Matching Garages:")
                    for garage in response['garages']:
                        with st.expander(f"🔧 {garage['GarageName']} - {garage['City']}"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("📍 **Location:**", garage['Location'])
                                st.write("🏙️ **City:**", garage['City'])
                                st.write("📮 **Postcode:**", garage['Postcode'])
                            with col2:
                                st.write("📞 **Phone:**", garage['Phone'])
                                st.write("📧 **Email:**", garage['Email'])
                                if 'Website' in garage and pd.notna(garage['Website']):
                                    st.write("🌐 **Website:**", garage['Website'])

if __name__ == "__main__":
    main()
