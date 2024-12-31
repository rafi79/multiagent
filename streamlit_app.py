import streamlit as st
import pandas as pd
import google.generativeai as genai
from openai import OpenAI
import requests
from typing import Dict, List, Optional
import re

# API Keys
OPENAI_KEY = "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"
GEMINI_KEY = "AIzaSyCcMZPrzP5me7Rl4pmAc1Nn5vUDSan5Q6E"
PERPLEXITY_KEY = "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"
GROQ_KEY = "groq-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"

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

class AIServices:
    """Manages different AI services"""
    def __init__(self):
        # Initialize Gemini
        try:
            genai.configure(api_key=GEMINI_KEY)
            self.gemini = genai.GenerativeModel('gemini-pro')
            st.session_state.gemini_available = True
        except Exception as e:
            st.warning("Gemini service not available")
            self.gemini = None
            st.session_state.gemini_available = False

        # Initialize OpenAI
        try:
            self.openai = OpenAI(api_key=OPENAI_KEY)
            st.session_state.dalle_available = True
            st.session_state.o1_available = True
        except Exception as e:
            st.warning("OpenAI services not available")
            self.openai = None
            st.session_state.dalle_available = False
            st.session_state.o1_available = False

    def location_search(self, query: str) -> str:
        """Use Gemini for location analysis"""
        if not self.gemini:
            return "Location service not available"
        try:
            response = self.gemini.generate_content(f"Extract location from: {query}")
            return response.text
        except Exception as e:
            return f"Error processing location: {str(e)}"

    def generate_image(self, prompt: str) -> Optional[str]:
        """Generate image using DALL-E"""
        if not self.openai:
            return None
        try:
            response = self.openai.images.generate(
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

    def get_analysis(self, query: str) -> str:
        """Get analysis using O1-mini"""
        if not self.openai:
            return "Analysis service not available"
        try:
            response = self.openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an automotive expert."},
                    {"role": "user", "content": query}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Analysis error: {str(e)}"

    def research_with_perplexity(self, query: str) -> str:
        """Research using Perplexity"""
        try:
            headers = {
                "Authorization": f"Bearer {PERPLEXITY_KEY}",
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
            return f"Research error: {str(e)}"

def create_welcome_section():
    """Create welcome section with examples"""
    st.title("🚗 Automotive Service Assistant")
    
    st.write("### Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎨 Find Paint Services"):
            st.session_state.example_query = "Where can I get my car painted in Bath?"
            
    with col2:
        if st.button("🔧 Design Custom Parts"):
            st.session_state.example_query = "I need a custom body kit visualization"
            
    with col3:
        if st.button("🔍 Find Best Garage"):
            st.session_state.example_query = "Which garage is best for BMW servicing?"

def process_garage_query(query: str, garage_data: pd.DataFrame, ai_services: AIServices) -> Dict:
    """Process garage-related query"""
    # Classify intent
    classifier = IntentClassifier()
    intents = classifier.classify_intent(query)
    
    response = {
        "message": "",
        "garages": [],
        "image_url": None
    }
    
    for intent in intents:
        if intent == "location":
            location = ai_services.location_search(query)
            filtered_garages = garage_data[
                garage_data['City'].str.lower().str.contains(location.lower())
            ]
            response["garages"].extend(filtered_garages.to_dict('records'))
            response["message"] += f"\nFound garages in {location}."
            
        elif intent == "paint":
            research = ai_services.research_with_perplexity(
                f"Research car painting services: {query}"
            )
            paint_garages = garage_data[
                garage_data['GarageName'].str.lower().str.contains('paint|body|finish', regex=True)
            ]
            response["garages"].extend(paint_garages.to_dict('records'))
            response["message"] += f"\n{research}"
            
        elif intent == "parts":
            if "visualization" in query.lower() or "design" in query.lower():
                image_url = ai_services.generate_image(
                    f"Professional automotive visualization: {query}"
                )
                response["image_url"] = image_url
                response["message"] += "\nGenerated visualization of your request."
                
        elif intent == "research":
            analysis = ai_services.get_analysis(query)
            response["message"] += f"\nExpert Analysis:\n{analysis}"
            
    return response

def main():
    # Page config
    st.set_page_config(page_title="Auto Service Assistant", layout="wide")
    
    # Initialize AI services
    if not st.session_state.initialized:
        st.session_state.ai_services = AIServices()
        st.session_state.initialized = True
    
    # Create welcome section
    create_welcome_section()
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        # Data upload
        uploaded_file = st.file_uploader("Upload Garage Database", type=['csv'])
        if uploaded_file is not None:
            st.session_state.garage_data = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(st.session_state.garage_data)} garages")
        
        # User information
        st.header("Your Contact Info")
        user_info = {
            "name": st.text_input("Name"),
            "email": st.text_input("Email"),
            "phone": st.text_input("Phone")
        }
        
        # Show AI service status
        st.header("AI Services Status")
        services = {
            "Gemini (Location)": st.session_state.get("gemini_available", False),
            "DALL-E (Design)": st.session_state.get("dalle_available", False),
            "O1-mini (Analysis)": st.session_state.get("o1_available", False),
            "Perplexity (Research)": True
        }
        for service, status in services.items():
            st.write(f"{'✅' if status else '❌'} {service}")
    
    # Chat interface
    st.write("### How can I help you today?")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            if message.get("garages"):
                for garage in message["garages"]:
                    with st.expander(f"{garage['GarageName']} - {garage['City']}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("📍 **Location:**", garage['Location'])
                            st.write("🏙️ **City:**", garage['City'])
                            st.write("📮 **Postcode:**", garage['Postcode'])
                        with col2:
                            st.write("📞 **Phone:**", garage['Phone'])
                            st.write("📧 **Email:**", garage['Email'])
                            if pd.notna(garage.get('Website')):
                                st.write("🌐 **Website:**", garage['Website'])
                                
                        if user_info["email"]:
                            if st.button("Contact Garage", key=garage['Email']):
                                st.info("Preparing to send email...")
                                # Add email sending logic here
            
            if message.get("image_url"):
                st.image(message["image_url"])
    
    # Chat input
    if "example_query" in st.session_state:
        query = st.chat_input("Type your question...", 
                            value=st.session_state.example_query)
        del st.session_state.example_query
    else:
        query = st.chat_input("Type your question...")
    
    if query:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": query})
        
        if st.session_state.garage_data is not None:
            with st.spinner("Processing your request..."):
                response = process_garage_query(
                    query,
                    st.session_state.garage_data,
                    st.session_state.ai_services
                )
                
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response["message"],
                    "garages": response["garages"],
                    "image_url": response.get("image_url")
                })
            
            st.rerun()
        else:
            st.warning("Please upload garage data to enable queries.")

if __name__ == "__main__":
    main()
