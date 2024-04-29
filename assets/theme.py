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

def update_time( time_display, start_time, end_time):
    global seconds_passed
    if 'seconds_passed' not in globals():
        seconds_passed = 0
    seconds_passed = seconds_passed + ( end_time - start_time)
    time_now = f"Time elapsed: {seconds_passed} seconds"
    time_display.empty()
    time_display.text(time_now)
    
