from tensorflow import keras
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf


with open("style.css") as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

st.write(
    """# Art Generation
Art Gen generates a fake portait using a GAN which learned from a dataset of more than 4000 art drawings.
You can also control the output by changing the values the noise vector by the Neural Network
"""
)

model = keras.models.load_model('generator1000.h5')
def generateFace(model):
    
    fig = plt.figure(figsize=(3, 3))
    rows = 4
    columns = 4
    
    for i in range(1,17):
        fig.add_subplot(rows, columns, i)
        noise = tf.random.normal([1, 100])
        generated_image = model(noise, training=False)
        plt.imshow((generated_image[0] * 127.5 + 127.5).numpy().astype('uint8'))
        plt.axis('off')
    return fig

if st.sidebar.button("Generate New art"):
    st.set_option("deprecation.showPyplotGlobalUse", False)
    st.pyplot(generateFace(model))
    



st.sidebar.markdown(
    "Get the Source Code [here](https://github.com/ChekrounMohammed/Streamlit-Generated-Art-GANs)",
    unsafe_allow_html=True,
)






