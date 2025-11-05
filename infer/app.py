import streamlit as st
import infer
from io import BytesIO

st.title("MIDI Humanizer")
st.write("An RNN based MIDI file humanizer")

uploaded_file = st.file_uploader("Choose a file")

# Check if a file was uploaded

if uploaded_file is not None:
    # Display file details
    st.write("Filename:", uploaded_file.name)
    st.write("File type:", uploaded_file.type)
    st.write("File size:", uploaded_file.size)

    # Read and display file content
    content = uploaded_file.read()

    if content[:4] == b'MThd':
        st.success("This is a valid MIDI file.")
        midi_stream = BytesIO(content)
        infer.process(midi_stream, "output.mid")

        with open("output.mid", "rb") as f:
            processed_content = f.read()

        st.download_button(
            label="Download processed MIDI file",
            data=processed_content,
            file_name="output.mid",
            mime="audio/midi"
        )
    else:
        st.error("This is not a MIDI file.")
