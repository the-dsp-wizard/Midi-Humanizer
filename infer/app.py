import streamlit as st
import infer
import wave
import numpy as np
import player
import mido
from io import BytesIO

st.title("Hello Streamlit!")
st.write("This is a simple Streamlit app.")

uploaded_file = st.file_uploader("Choose a file")

amnt = st.slider("Select Humanization Amount", 0.0, 1.0, 0.5)
release = st.slider("Select Synth Release Time", 0.1, 5.0, 0.5)

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

        processed = infer.process(midi_stream, "output.mid", amnt)

        st.download_button(
            label="Download processed MIDI file",
            data=processed,
            file_name="output.mid",
            mime="audio/midi"
        )

        mid = mido.MidiFile(file=BytesIO(processed))

        fs = 44100
        synth = (player.synth(fs, release, mid) * 32767).astype(np.int16)

        buffer = BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(fs)
            wf.writeframes(synth.tobytes())

        # Reset buffer position
        buffer.seek(0)

        # Play in Streamlit
        st.audio(buffer, format="audio/wav")

    else:
        st.error("This is not a MIDI file.")
