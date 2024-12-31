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
    st.session_state.garage_data = None
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

class AutoServiceAgent:
    def __init__(self):
        # Initialize API clients
        self.openai = OpenAI(api_key="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")
        genai.configure(api_key="AIzaSyCcMZPrzP5me7Rl4pmAc1Nn5vUDSan5Q6E")
        self.gemini = genai.GenerativeModel('gemini-pro')
        self.intent_classifier = IntentClassifier()
        
    def process_query(self, query: str, garage_data: pd.DataFrame) -> Dict:
        """Process user query and route to appropriate agent"""
        intents = self.intent_classifier.classify_intent(query)
        response = {"message": "", "garages": [], "visualizations": None}
        
        for intent in intents:
            if intent == "location":
                location_info = self._process_location(query, garage_data)
                response["message"] += f"\n\n{location_info['message']}"
                response["garages"].extend(location_info["garages"])
                
            elif intent == "paint":
                paint_info = self._process_paint(query, garage_data)
                response["message"] += f"\n\n{paint_info['message']}"
                response["garages"].extend(paint_info["garages"])
                
            elif intent == "parts":
                parts_info = self._process_parts(query)
                response["message"] += f"\n\n{parts_info['message']}"
                response["visualizations"] = parts_info.get("image_url")
                
            elif intent == "research":
                research_info = self._process_research(query, garage_data)
                response["message"] += f"\n\n{research_info['message']}"
                response["garages"].extend(research_info["garages"])
                
        return response
    
    def _process_location(self, query: str, garage_data: pd.DataFrame) -> Dict:
        """Use Gemini for location-based queries"""
        try:
            # Extract location from query
            response = self.gemini.generate_content(
                f"Extract the city or location from: {query}"
            )
            location = response.text.strip()
            
            # Filter garages by location
            relevant_garages = garage_data[
                garage_data['City'].str.lower().str.contains(location.lower())
            ]
            
            return {
                "message": f"Found these garages in {location}:",
                "garages": relevant_garages.to_dict('records')
            }
        except Exception as e:
            st.error(f"Location processing error: {str(e)}")
            return {"message": "Error processing location", "garages": []}
            
    def _process_paint(self, query: str, garage_data: pd.DataFrame) -> Dict:
        """Use Perplexity to research paint services"""
        try:
            headers = {
                "Authorization": f"Bearer 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",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "llama-3.1-sonar-small-128k-online",
                "messages": [
                    {
                        "role": "user",
                        "content": f"Research car painting services for: {query}"
                    }
                ]
            }
            
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers=headers,
                json=payload
            ).json()
            
            analysis = response['choices'][0]['message']['content']
            
            # Filter garages based on analysis
            relevant_garages = garage_data[
                garage_data['GarageName'].str.lower().str.contains('paint|body|finish', regex=True)
            ]
            
            return {
                "message": f"Paint Service Analysis:\n{analysis}",
                "garages": relevant_garages.to_dict('records')
            }
        except Exception as e:
            st.error(f"Paint research error: {str(e)}")
            return {"message": "Error researching paint services", "garages": []}
            
    def _process_parts(self, query: str) -> Dict:
        """Use DALL-E for parts visualization"""
        try:
            # Generate part visualization
            response = self.openai.images.generate(
                model="dall-e-3",
                prompt=f"Professional automotive visualization: {query}",
                size="1024x1024",
                quality="standard",
                n=1
            )
            
            return {
                "message": "Generated visualization of the requested part:",
                "image_url": response.data[0].url
            }
        except Exception as e:
            st.error(f"Parts visualization error: {str(e)}")
            return {"message": "Error generating part visualization"}
            
    def _process_research(self, query: str, garage_data: pd.DataFrame) -> Dict:
        """Use O1-mini for garage recommendations"""
        try:
            # Get analysis from O1-mini
            response = self.openai.chat.completions.create(
                model="o1-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an automotive expert helping recommend garages."
                    },
                    {
                        "role": "user",
                        "content": f"Analyze this request and recommend garage criteria: {query}"
                    }
                ]
            )
            
            analysis = response.choices[0].message.content
            
            # Filter garages based on analysis
            # This is a simple filter - you can make it more sophisticated
            relevant_garages = garage_data.head(3)  # Placeholder
            
            return {
                "message": f"Expert Analysis:\n{analysis}",
                "garages": relevant_garages.to_dict('records')
            }
        except Exception as e:
            st.error(f"Research error: {str(e)}")
            return {"message": "Error analyzing request", "garages": []}

def send_email_to_garage(garage: dict, user_query: str, user_contact: dict):
    """Send email using Groq/Mixtral"""
    try:
        llm = ChatGroq(
            model="mixtral-8x7b-32768",
            api_key="groq-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"
        )
        
        tool_set = CompostoToolSet()
        tools = tool_set.get_tools(App.Gmail)
        
        user_prompt = f"""
        Send an email to {garage['Email']}
        Subject: Service Inquiry
        
        Query: {user_query}
        
        Contact Information:
        Name: {user_contact['name']}
        Email: {user_contact['email']}
        Phone: {user_contact['phone']}
        """
        
        response = agent_executor.invoke(
            llm=llm,
            user_prompt=user_prompt,
            tools=tools
        )
        
        return True
    except Exception as e:
        st.error(f"Email error: {str(e)}")
        return False

def main():
    st.title("🚗 Auto Service Assistant")
    
    # Initialize agent if not already done
    if not st.session_state.initialized:
        st.session_state.agent = AutoServiceAgent()
        st.session_state.initialized = True
    
    # Sidebar for data upload
    with st.sidebar:
        uploaded_file = st.file_uploader("Upload Garage CSV", type=['csv'])
        if uploaded_file is not None:
            st.session_state.garage_data = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(st.session_state.garage_data)} garages")
            
        # Contact information
        st.header("Your Contact Info")
        user_contact = {
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
                        if user_contact["email"]:
                            if st.button(f"Contact {garage['GarageName']}", key=garage['Email']):
                                if send_email_to_garage(garage, message["content"], user_contact):
                                    st.success("Email sent!")
            if message.get("image_url"):
                st.image(message["image_url"])
    
    # Chat input
    if query := st.chat_input("Ask about services, locations, or specific needs..."):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": query})
        
        # Process query and get response
        if st.session_state.garage_data is not None:
            response = st.session_state.agent.process_query(query, st.session_state.garage_data)
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response["message"],
                "garages": response["garages"],
                "image_url": response.get("visualizations")
            })
            
            # Force refresh
            st.rerun()
        else:
            st.warning("Please upload garage data to enable queries.")

if __name__ == "__main__":
    main()
