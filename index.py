import streamlit as st

# Placeholder functions for model script generation
def generate_tinyllama_script(prompt):
    return f"TinyLlama Output: \nOPERATOR: 911, what's your emergency?\nCALLER: {prompt} There's a fire at 123 Elm Street!"

def generate_nanogpt_script(prompt):
    return f"NanoGPT Output: \nOPERATOR: 911, how can I help?\nCALLER: {prompt} I need help at 456 Oak Avenue!"

def generate_custom_script(prompt):
    return f"Custom Model Output: \nOPERATOR: 911, what's the issue?\nCALLER: {prompt} Emergency at 789 Pine Road!"

# Streamlit app layout
st.title("Emergency Call Script Generator")
st.write("Select a model and enter a prompt to generate realistic emergency call scripts.")

# Sidebar for model selection
with st.sidebar:
    st.header("Model Selection")
    model_choice = st.selectbox(
        "Choose a Model",
        ["TinyLlama", "NanoGPT", "Custom Model"]
    )
    st.write(f"Selected Model: **{model_choice}**")
    st.write("Use the main panel to input a prompt and view generated scripts.")

# Main panel for prompt input and output
st.subheader("Generate Script")
prompt = st.text_area("Enter a prompt", height=100)
if st.button("Generate"):
    if prompt.strip():
        with st.spinner("Generating script..."):
            if model_choice == "TinyLlama":
                output = generate_tinyllama_script(prompt)
            elif model_choice == "NanoGPT":
                output = generate_nanogpt_script(prompt)
            else:
                output = generate_custom_script(prompt)
            st.success("Script generated!")
            st.markdown("### Generated Script")
            st.write(output)
    else:
        st.error("Please enter a valid prompt.")

# Footer
st.markdown("---")
st.write("Developed for Merseyside Fire and Rescue Training")