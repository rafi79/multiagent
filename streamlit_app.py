import streamlit as st
import pandas as pd
import google.generativeai as genai
from openai import OpenAI
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

class IntentClassifier:
    """Classifies user intent to route to appropriate agent"""
    def __init__(self):
        self.intent_patterns = {
            "location": r"(where|location|find|near|closest|map)",
            "paint": r"(paint|painting|color|bodywork|finish)",
            "parts": r"(part|parts|component|replacement|fix)",
            "research": r"(which|recommend|best|compare|review|suggest)",
            "contact": r"(contact|email|send|message|book|appointment)"
        }
    
    def classify_intent(self, query: str) -> List[str]:
        query = query.lower()
        intents = []
        for intent, pattern in self.intent_patterns.items():
            if re.search(pattern, query):
                intents.append(intent)
        return intents or ["general"]

class ServiceHandler:
    """Handles different service requests"""
    
    def __init__(self):
        # Initialize APIs
        try:
            genai.configure(api_key="AIzaSyCcMZPrzP5me7Rl4pmAc1Nn5vUDSan5Q6E")
            self.gemini = genai.GenerativeModel('gemini-pro')
        except Exception as e:
            st.warning("Gemini API initialization failed")
            self.gemini = None
            
    def process_location_query(self, query: str, garage_data: pd.DataFrame) -> Dict:
        """Process location-based queries"""
        try:
            if not self.gemini:
                return {"message": "Location service unavailable", "garages": []}
                
            # Extract location
            response = self.gemini.generate_content(f"Extract the city name from: {query}")
            location = response.text.strip()
            
            # Filter garages
            relevant_garages = garage_data[
                garage_data['City'].str.lower().str.contains(location.lower())
            ]
            
            return {
                "message": f"Found these garages in {location}:",
                "garages": relevant_garages.to_dict('records')
            }
        except Exception as e:
            return {"message": f"Error processing location: {str(e)}", "garages": []}
    
    def process_paint_query(self, query: str, garage_data: pd.DataFrame) -> Dict:
        """Process paint service queries using Perplexity"""
        try:
            headers = {
                "Authorization": "Bearer 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",
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
            
            if response.status_code == 200:
                result = response.json()
                analysis = result['choices'][0]['message']['content']
                
                # Filter garages
                relevant_garages = garage_data[
                    garage_data['GarageName'].str.lower().str.contains('paint|body|finish', regex=True)
                ]
                
                return {
                    "message": f"Paint Service Analysis:\n{analysis}",
                    "garages": relevant_garages.to_dict('records')
                }
            else:
                return {"message": "Error getting paint service information", "garages": []}
                
        except Exception as e:
            return {"message": f"Error researching paint services: {str(e)}", "garages": []}

class QueryProcessor:
    """Processes user queries and coordinates responses"""
    
    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.service_handler = ServiceHandler()
        
    def process_query(self, query: str, garage_data: pd.DataFrame) -> Dict:
        """Process user query and return appropriate response"""
        intents = self.intent_classifier.classify_intent(query)
        response = {"message": "", "garages": [], "visualizations": None}
        
        for intent in intents:
            if intent == "location":
                location_info = self.service_handler.process_location_query(query, garage_data)
                response["message"] += f"\n\n{location_info['message']}"
                response["garages"].extend(location_info["garages"])
                
            elif intent == "paint":
                paint_info = self.service_handler.process_paint_query(query, garage_data)
                response["message"] += f"\n\n{paint_info['message']}"
                response["garages"].extend(paint_info["garages"])
                
        return response

def create_email_content(garage: dict, query: str, user_info: dict) -> str:
    """Create email content for garage communication"""
    return f"""
    Subject: Service Inquiry
    
    Dear {garage['GarageName']},
    
    I am writing to inquire about your services.
    
    Query: {query}
    
    Contact Information:
    Name: {user_info.get('name', 'Not provided')}
    Email: {user_info.get('email', 'Not provided')}
    Phone: {user_info.get('phone', 'Not provided')}
    
    Best regards,
    {user_info.get('name', 'A potential customer')}
    """

def main():
    st.title("🚗 Auto Service Assistant")
    
    # Initialize processor if not already done
    if not st.session_state.initialized:
        st.session_state.processor = QueryProcessor()
        st.session_state.initialized = True
    
    # Sidebar for data upload and user info
    with st.sidebar:
        uploaded_file = st.file_uploader("Upload Garage CSV", type=['csv'])
        if uploaded_file is not None:
            st.session_state.garage_data = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(st.session_state.garage_data)} garages")
            
        # Contact information
        st.header("Your Contact Info")
        user_info = {
            "name": st.text_input("Your Name"),
            "email": st.text_input("Your Email"),
            "phone": st.text_input("Your Phone")
        }
    
    # Main chat interface
    st.write("How can I help you today?")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message.get("garages"):
                for garage in message["garages"]:
                    with st.expander(f"{garage['GarageName']} - {garage['City']}"):
                        st.write(f"**Location:** {garage['Location']}")
                        st.write(f"**Contact:** {garage['Phone']}")
                        st.write(f"**Email:** {garage['Email']}")
                        
                        if user_info["email"]:
                            email_content = create_email_content(
                                garage, 
                                message["content"], 
                                user_info
                            )
                            if st.button(f"Contact {garage['GarageName']}", key=garage['Email']):
                                st.info("Email content preview:")
                                st.text(email_content)
    
    # Chat input
    if query := st.chat_input("Ask about services, locations, or specific needs..."):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": query})
        
        # Process query
        if st.session_state.garage_data is not None:
            with st.spinner("Processing your request..."):
                response = st.session_state.processor.process_query(
                    query, 
                    st.session_state.garage_data
                )
                
                # Add assistant response to chat history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response["message"],
                    "garages": response["garages"]
                })
            
            # Force refresh
            st.rerun()
        else:
            st.warning("Please upload garage data to enable queries.")

if __name__ == "__main__":
    main()
