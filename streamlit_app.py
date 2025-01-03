import streamlit as st
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Auto Service Assistant",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "garage_data" not in st.session_state:
    st.session_state.garage_data = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def main():
    st.title("🚗 Automotive Service Assistant")
    
    # Custom CSS
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
        }
        </style>
    """, unsafe_allow_html=True)

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
                
                # Data preview
                with st.expander("Preview Data"):
                    st.dataframe(st.session_state.garage_data.head())
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")

        st.header("Quick Actions")
        if st.button("🎨 Paint Services"):
            st.session_state.current_query = "Find paint services"
        if st.button("🔧 Best Garages"):
            st.session_state.current_query = "Best garage recommendations"
        if st.button("📍 Nearby Garages"):
            st.session_state.current_query = "Find garages near me"

    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Ask about services")
        st.markdown("""
        You can ask about:
        - 🔍 Finding specific services in your area
        - 🎨 Paint and bodywork services
        - ⭐ Garage recommendations
        - 📍 Location-based searches
        """)

    with col2:
        st.info("💡 Upload a garage database to get started!")

    # Search interface
    st.subheader("Search for Services")
    search_col1, search_col2 = st.columns([3, 1])
    
    with search_col1:
        query = st.text_input(
            "Your question:",
            placeholder="e.g., Find garages in London specializing in paint work"
        )
    
    with search_col2:
        if st.button("🔍 Search", type="primary", use_container_width=True):
            if st.session_state.garage_data is None:
                st.error("Please upload a garage database first!")
            else:
                st.session_state.chat_history.append({"role": "user", "content": query})
                with st.spinner("Processing your request..."):
                    # Placeholder for future response processing
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": "This is a placeholder response. Add your processing logic here."
                    })

    # Chat history display
    st.header("Conversation History")
    for message in st.session_state.chat_history:
        with st.container():
            if message["role"] == "user":
                st.markdown(f"🙋 **You:** {message['content']}")
            else:
                st.markdown(f"🤖 **Assistant:** {message['content']}")

    # Footer
    st.markdown("---")
    st.markdown("### Need Help?")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("📋 **Upload Data Format**")
        st.markdown("Required columns: GarageName, Location, City, Services")
    with col2:
        st.markdown("🔍 **Search Tips**")
        st.markdown("Be specific about location and service needs")
    with col3:
        st.markdown("💡 **Quick Links**")
        st.markdown("- Search by service\n- Find nearest garage\n- Compare services")

if __name__ == "__main__":
    main()
