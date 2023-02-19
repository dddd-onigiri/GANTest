import streamlit as st
from PIL import Image
from io import BytesIO
import requests
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

st.title('Image Generation App')

# テキストを入力
text_input = st.text_input('Enter text', 'happy')

# GANモデルをロード
model = load_model('generator_model.h5', compile=False)

# 画像生成関数
def generate_image(text):
    # テキストをベクトル化
    vectorized_text = np.array([text])

    # 画像生成
    generated_image = model.predict(vectorized_text)

    # 画像を配列に変換
    generated_image_array = np.squeeze(generated_image, axis=0)
    generated_image_array = (generated_image_array * 127.5 + 127.5).astype(np.uint8)

    # 配列を画像に変換
    generated_image = Image.fromarray(generated_image_array)

    return generated_image

# 画像を表示
if text_input:
    generated_image = generate_image(text_input)
    st.image(generated_image, caption=f'Generated image for "{text_input}"', use_column_width=True)
