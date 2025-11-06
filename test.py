import streamlit as st

# --- CENTER THE TITLE ---
st.markdown("<h1 style='text-align: center;'>MUSE AI</h1>", unsafe_allow_html=True)


# --- CENTER THE TEXT ---
st.markdown("<p style='text-align: center;'>Click the button below to scan the object.</p>", unsafe_allow_html=True)


# --- CENTER THE BUTTON ---
# 1. Create three columns
#    The first and third columns act as "spacers"
#    [2, 1, 2] means the middle column is 1/5th of the width
#    and centered. Adjust the numbers to change the spacing.
col1, mid_col, col3 = st.columns([2, 1, 2])

# 2. Put the button in the middle column
with mid_col:
    # We check the button press inside the 'with' block
    if st.button("Click me"):
        st.balloons() # Add a fun action
