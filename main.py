import os
from PIL import Image
import streamlit as st
from streamlit_option_menu import option_menu
from gemini_utility import (load_swag_ai_model,
                             swag_ai_response,
                             swag_ai_vision_response,
                             swag_ai_embeddings_response)
import pdfplumber  # Import pdfplumber for PDF processing

working_dir = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="Swag AI",
    page_icon="🧠",
    layout="centered",
)

with st.sidebar:
    selected = option_menu('Swag AI',
                           ['ChatBot',
                            'Image Captioning',
                            'Embed Text',
                            'Ask Me Anything'],
                           menu_icon='robot', icons=['chat-dots-fill', 'image-fill', 'textarea-t', 'patch-question-fill'],
                           default_index=0
                           )

# Chatbot page
if selected == 'ChatBot':
    model = load_swag_ai_model()

    # Initialize chat session in Streamlit if not already present
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = model.start_chat(history=[])

    # Display the chatbot's title on the page
    st.title("🤖 ChatBot")

    # Display the chat history
    for message in st.session_state.chat_session.history:
        with st.chat_message(translate_role_for_streamlit(message.role)):
            st.markdown(message.parts[0].text)

    # Input field for user's message
    user_prompt = st.chat_input("Ask Swag AI...")
    if user_prompt:
        # Add user's message to chat and display it
        st.chat_message("user").markdown(user_prompt)

        # Send user's message to Swag AI and get the response
        swag_ai_response = st.session_state.chat_session.send_message(user_prompt)

        # Display Swag AI's response
        with st.chat_message("assistant"):
            st.markdown(swag_ai_response.text)

# Image captioning page
if selected == "Image Captioning":
    st.title("📷 Snap Narrate")

    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if st.button("Generate Caption") and uploaded_image is not None:
        try:
            image = Image.open(uploaded_image)

            col1, col2 = st.columns(2)

            with col1:
                resized_img = image.resize((800, 500))
                st.image(resized_img)

            default_prompt = "Write a short caption for this image"

            # Get the caption of the image from the Swag AI Vision model
            caption = swag_ai_vision_response(default_prompt, image)

            with col2:
                st.info(caption)

        except Exception as e:
            st.error(f"Error processing the image: {e}")

# Embed Text page with PDF upload functionality
if selected == "Embed Text":
    st.title("🔡 Embed Text from PDF")

    # File uploader for PDFs
    uploaded_pdf = st.file_uploader("Upload a PDF...", type=["pdf"])

    if uploaded_pdf is not None:
        with pdfplumber.open(uploaded_pdf) as pdf:
            full_text = ""
            images = []

            for page in pdf.pages:
                # Extract text from the page
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"

                # Extract images from the page
                page_images = page.images
                for img in page_images:
                    # Extract image and convert to PIL format
                    image = page.to_image()
                    pil_image = image.original
                    images.append(pil_image)

        # Display extracted text
        st.text_area("Extracted Text", full_text, height=300)

        # Display images
        if images:
            st.subheader("Images Extracted from PDF:")
            for idx, img in enumerate(images):
                st.image(img, caption=f"Image {idx + 1}", use_column_width=True)

        # Input for user questions
        user_question = st.text_area("Ask a question about the PDF...")

        if st.button("Get Response"):
            if user_question:
                # Call your AI model with the question and the extracted text
                response = swag_ai_embeddings_response(f"{full_text}\n\nQuestion: {user_question}")
                st.markdown(response)

# Ask me anything model
if selected == "Ask Me Anything":
    st.title("❓ Ask Me a Question")

    # Text box to enter prompt
    user_prompt = st.text_area(label='', placeholder="Ask me anything...")

    if st.button("Get Response"):
        response = swag_ai_response(user_prompt)
        st.markdown(response)
