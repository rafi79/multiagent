import streamlit as st
import pandas as pd
import google.generativeai as genai
import requests
from typing import Dict, List, Optional
import re

# Initialize session state
if "initialized" not in st.session_state:
    st.session_state.initialized = False
if "garage_data" not in st.session_state:
    st.session_state.garage_data = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_query" not in st.session_state:
    st.session_state.current_query = ""

class GarageAssistant:
    def __init__(self):
        # Initialize Gemini
        try:
            genai.configure(api_key="AIzaSyCcMZPrzP5me7Rl4pmAc1Nn5vUDSan5Q6E")
            self.gemini = genai.GenerativeModel('gemini-pro')
        except Exception as e:
            st.warning("Gemini service not available")
            self.gemini = None

    def process_query(self, query: str, garage_data: pd.DataFrame) -> Dict:
        """Process user query and return response"""
        response = {
            "message": "",
            "garages": [],
            "type": "text"
        }

        # Analyze query intent
        if any(word in query.lower() for word in ['where', 'location', 'find', 'near']):
            # Location-based search
            location_info = self._handle_location_query(query, garage_data)
            response.update(location_info)

        elif any(word in query.lower() for word in ['paint', 'painting', 'bodywork']):
            # Paint service search
            paint_info = self._handle_paint_query(query, garage_data)
            response.update(paint_info)

        elif any(word in query.lower() for word in ['best', 'recommend', 'which']):
            # Recommendation search
            rec_info = self._handle_recommendation_query(query, garage_data)
            response.update(rec_info)

        return response

    def _handle_location_query(self, query: str, garage_data: pd.DataFrame) -> Dict:
        """Handle location-based queries using Gemini"""
        try:
            if self.gemini:
                # Extract location from query
                response = self.gemini.generate_content(f"Extract just the city name from: {query}")
                location = response.text.strip().lower()
                
                # Filter garages by location
                relevant_garages = garage_data[
                    garage_data['City'].str.lower().str.contains(location)
                ]
                
                return {
                    "message": f"Found these garages in {location.title()}:",
                    "garages": relevant_garages.to_dict('records'),
                    "type": "location"
                }
            else:
                return {
                    "message": "Location service currently unavailable. Please try a direct search.",
                    "garages": [],
                    "type": "error"
                }
        except Exception as e:
            return {
                "message": f"Error processing location query: {str(e)}",
                "garages": [],
                "type": "error"
            }

    def _handle_paint_query(self, query: str, garage_data: pd.DataFrame) -> Dict:
        """Handle paint service queries using Perplexity"""
        try:
            # Research paint services
            headers = {
                "Authorization": f"Bearer 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",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "llama-3.1-sonar-small-128k-online",
                "messages": [
                    {"role": "user", "content": f"Research car painting services for: {query}"}
                ]
            }

            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers=headers,
                json=payload
            )
            
            analysis = response.json()['choices'][0]['message']['content']
            
            # Filter paint-related garages
            paint_garages = garage_data[
                garage_data['GarageName'].str.lower().str.contains('paint|body|finish', regex=True)
            ]
            
            return {
                "message": f"Paint Service Analysis:\n{analysis}",
                "garages": paint_garages.to_dict('records'),
                "type": "paint"
            }
        except Exception as e:
            return {
                "message": f"Error researching paint services: {str(e)}",
                "garages": [],
                "type": "error"
            }

    def _handle_recommendation_query(self, query: str, garage_data: pd.DataFrame) -> Dict:
        """Handle recommendation queries"""
        try:
            # Use Perplexity for research
            headers = {
                "Authorization": f"Bearer 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",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "llama-3.1-sonar-small-128k-online",
                "messages": [
                    {"role": "user", "content": f"Analyze this automotive service query and provide recommendations: {query}"}
                ]
            }

            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers=headers,
                json=payload
            )
            
            analysis = response.json()['choices'][0]['message']['content']
            
            # Extract key terms from query for filtering
            search_terms = re.findall(r'\b\w+\b', query.lower())
            relevant_garages = garage_data[
                garage_data['GarageName'].str.lower().str.contains('|'.join(search_terms), regex=True)
            ]
            
            return {
                "message": f"Expert Analysis:\n{analysis}",
                "garages": relevant_garages.to_dict('records'),
                "type": "recommendation"
            }
        except Exception as e:
            return {
                "message": f"Error generating recommendations: {str(e)}",
                "garages": [],
                "type": "error"
            }

def main():
    st.title("🚗 Automotive Service Assistant")

    # Initialize assistant
    if not st.session_state.initialized:
        st.session_state.assistant = GarageAssistant()
        st.session_state.initialized = True

    # Sidebar
    with st.sidebar:
        st.header("Data Upload")
        uploaded_file = st.file_uploader("Upload Garage Database", type=['csv'])
        if uploaded_file is not None:
            st.session_state.garage_data = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(st.session_state.garage_data)} garages")

        st.header("Quick Questions")
        if st.button("🎨 Find Paint Services"):
            st.session_state.current_query = "Where can I get my car painted?"
        if st.button("🔍 Best Garage Recommendation"):
            st.session_state.current_query = "Which is the best garage for servicing?"
        if st.button("📍 Find Nearby Garages"):
            st.session_state.current_query = "Find garages near me"

    # Main interface
    st.write("### Ask me about:")
    st.write("- Finding specific services in your area")
    st.write("- Paint and bodywork services")
    st.write("- Garage recommendations")

    # Query input
    query = st.text_input("Your question:", value=st.session_state.current_query)
    st.session_state.current_query = ""  # Clear after use

    if query and st.button("Send"):
        # Add user query to chat history
        st.session_state.chat_history.append({"role": "user", "content": query})

        # Process query if garage data is available
        if st.session_state.garage_data is not None:
            with st.spinner("Processing your request..."):
                response = st.session_state.assistant.process_query(
                    query,
                    st.session_state.garage_data
                )

                # Add response to chat history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response["message"],
                    "garages": response["garages"]
                })

    # Display chat history
    for message in st.session_state.chat_history:
        with st.container():
            if message["role"] == "user":
                st.write(f"🙋 **You:** {message['content']}")
            else:
                st.write(f"🤖 **Assistant:** {message['content']}")
                
                if message.get("garages"):
                    st.write("**Matching Garages:**")
                    for garage in message["garages"]:
                        with st.expander(f"🏢 {garage['GarageName']} - {garage['City']}"):
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
    st.set_page_config(page_title="Auto Service Assistant", layout="wide")
    main()
