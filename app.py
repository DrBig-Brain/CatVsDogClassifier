import streamlit
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

model = load_model('model.keras')

classes = ['cat','dog']

streamlit.title('CNN Cat Dog Classifier')
streamlit.subheader("upload an Image")

upload_file = streamlit.file_uploader("choose and image ...",type = ['jpg','jpeg','png'])

if upload_file is not None:
    img = Image.open(upload_file)
    streamlit.image(img, caption = 'upload_image',use_column_width=True)

    if streamlit.button('predict'):
        img_resized = img.resize((256,256))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array,axis = 0)/255.0

        prediction = model.predict(img_array)
        if prediction>=0.5:
            p=1
        else:
            p=0
        predicted_class = classes[p]

        streamlit.success(f'prediction:{predicted_class},{prediction}')