import streamlit as st
import pandas as pd
import json
import os
from lida import Manager, TextGenerationConfig, llm
from tempfile import NamedTemporaryFile
import matplotlib as plt
import base64
from PIL import Image
import io

# Function to categorize titles
def categorize_title(title):
    title_lower = title.lower()

    # Excluding 'New Tab' from categorization
    if title_lower == 'new tab':
        return None

    # Communication
    if any(keyword in title_lower for keyword in ['email', 'outlook', 'whatsapp', 'chat', 'gmail', 'sign in', 'login']):
        return 'Communication'
    
    # Professional
    elif any(keyword in title_lower for keyword in ['linkedin', 'jobs', 'career', 'recruitment', 'professional']):
        return 'Professional'
    
    # Educational
    elif any(keyword in title_lower for keyword in ['course', 'colaboratory', 'learn', 'education', 'school', 'university', 'college', 'class', 'online test']):
        return 'Educational'
    
    # Financial
    elif any(keyword in title_lower for keyword in ['bank', 'finance', 'gst', 'payment', 'tax', 'economy']):
        return 'Financial'
    
    # Entertainment
    elif any(keyword in title_lower for keyword in ['disney+', 'hotstar', 'netflix', 'youtube', 'movie', 'tv', 'video', 'stream']):
        return 'Entertainment'
    
    # Productivity
    elif any(keyword in title_lower for keyword in ['workday', 'onedrive', 'dashboard', 'planner', 'chatgpt', 'api', 'tool', 'manager', 'software']):
        return 'Productivity'

    # Navigation
    elif 'google maps' in title_lower:
        return 'Navigation'
    
    # Others
    else:
        return 'Others'
    
def load_data(uploaded_file):
    data = json.load(uploaded_file)
    browser_history_df = pd.DataFrame(data["Browser History"])

    if 'time_usec' in browser_history_df.columns:
        browser_history_df['time'] = pd.to_datetime(browser_history_df['time_usec'], unit='us')
        browser_history_df.drop('time_usec', axis=1, inplace=True)

    # Apply the categorize_title function to create a new column
    browser_history_df['category'] = browser_history_df['title'].apply(categorize_title)
    
    return browser_history_df

def main():
    st.title('👩🏻‍💻 Browser History Analysis with AI')

    uploaded_file = st.file_uploader("Upload your Google History JSON file", type="json")
    api_key = st.text_input("Enter your OpenAI API key:", type="password")

    if uploaded_file and api_key:
        # Set the OpenAI API key
        os.environ["OPENAI_API_KEY"] = api_key

        # Process the uploaded file
        browser_history_df = load_data(uploaded_file)

        # Create a temporary CSV file from the DataFrame
        with NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            browser_history_df.to_csv(tmp_file.name, index=False)

            # Initialize LIDA
            text_gen = llm("openai")
            lida = Manager(text_gen=text_gen)

            # Generate summary and goals using LIDA
            summary = lida.summarize(tmp_file.name)
            goals = lida.goals(summary, n=1)  # exploratory data analysis

            st.subheader("Data Analysis Goals")
            for i, goal in enumerate(goals):
                st.write(f"Goal {i+1}: {goal}")

            # Allow users to select a goal
            goal_index = st.selectbox("Select a goal to visualize", range(len(goals)))
            library = "seaborn"
            textgen_config = TextGenerationConfig(n=1, temperature=0.2, use_cache=True)

            # Generate visualizations
        visualizations = lida.visualize(
            summary=summary, 
            goal=goals[goal_index], 
            textgen_config=textgen_config, 
            library=library
        )

        # Create titles for each visualization
        viz_titles = [f'Visualization {i + 1}' for i in range(len(visualizations))]

        # User selects a visualization
        selected_viz_title = st.selectbox('Choose a visualization', options=viz_titles, index=0)
        selected_viz = visualizations[viz_titles.index(selected_viz_title)]

        # Display the selected visualization
        if selected_viz.raster:
            # Decode and display raster image
            imgdata = base64.b64decode(selected_viz.raster)
            img = Image.open(io.BytesIO(imgdata))
            st.image(img, caption=selected_viz_title, use_column_width=True)
        else:
            st.error("Visualization format not supported.")

        # Display the code for the selected visualization
        st.write("### Visualization Code")
        st.code(selected_viz.code)

        # Chatbot feature for user queries
        st.subheader("Ask for a Custom Visualization")
        user_query = st.text_input("Enter your query:")
        if st.button("Generate Visualization"):
            if user_query:
                # Process user query and generate visualization
                query_visualizations = lida.visualize(
                    summary=summary, 
                    goal=user_query, 
                    textgen_config=textgen_config, 
                    library=library
                )

                # Display the visualization
                if query_visualizations:
                    query_viz = query_visualizations[0]
                    if query_viz.raster:
                        imgdata = base64.b64decode(query_viz.raster)
                        img = Image.open(io.BytesIO(imgdata))
                        st.image(img, caption="Your Custom Visualization", use_column_width=True)
                    else:
                        st.error("Custom visualization format not supported.")
                    
                    # Display the code for the custom visualization
                    st.write("### Visualization Code")
                    st.code(query_viz.code)
                else:
                    st.error("No visualization generated for the query.")
            else:
                st.error("Please enter a query to generate a visualization.")

        
if __name__ == "__main__":
    main()
