import streamlit as st


def set_custom_theme(title):
    st.write(f"<h2 style='text-align: center; color: red; font-size: 38px; font-weight: bold; font-family: Times New Roman, sans-serif ;'>{title}</h2>", unsafe_allow_html=True)
    page_bg_img = '''
        <style>
        .stApp {
            background-image: url("https://img.freepik.com/free-vector/blue-curve-background_53876-113112.jpg");
            background-size: 100vw 100vh; 
            background-position: center;  
            background-repeat: no-repeat;
        }
    </style>
    '''

    st.markdown(page_bg_img, unsafe_allow_html=True)
    
