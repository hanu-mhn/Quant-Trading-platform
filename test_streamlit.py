import streamlit as st

st.title("Simple Test App")
st.write("If you can see this, Streamlit is working correctly!")

# Add a simple interactive element
if st.button("Click me"):
    st.success("Button clicked!")
    
# Show some simple data
import pandas as pd
import numpy as np

data = pd.DataFrame({
    'A': np.random.rand(10),
    'B': np.random.rand(10)
})

st.write("Sample dataframe:")
st.dataframe(data)

st.write("Sample chart:")
st.line_chart(data)
