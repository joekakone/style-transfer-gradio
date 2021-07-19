# coding : utf-8

'''
Style Transfer
Credits: https://www.tensorflow.org/tutorials/generative/style_transfer
Gradio: https://gradio.app
'''

import os
import numpy as np
from PIL import Image
import gradio as gr
import tensorflow as tf
import tensorflow_hub as hub


FILE_PATH = 'temp.png'


def preprocess_input(image):
	max_dim = 512
	image = Image.fromarray(image)
	# save file temporarily
	image.save(FILE_PATH)

	img = tf.io.read_file(FILE_PATH)
	# remove file
	os.remove(FILE_PATH)
	img = tf.image.decode_image(img, channels=3)
	img = tf.image.convert_image_dtype(img, tf.float32)

	shape = tf.cast(tf.shape(img)[:-1], tf.float32)
	long_dim = max(shape)
	scale = max_dim / long_dim

	new_shape = tf.cast(shape * scale, tf.int32)

	img = tf.image.resize(img, new_shape)
	img = img[tf.newaxis, :]

	return img

def style_transfer(input_content_image, input_style_image):
	content_image = preprocess_input(input_content_image)
	style_image = preprocess_input(input_style_image)

	stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]
	stylized_image = np.array(stylized_image)
	stylized_image = np.reshape(stylized_image, stylized_image.shape[1:])

	return stylized_image


# retreive model from tfhub
print("Download model from Tensorflow Hub...")
hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# inputs
input1 = gr.inputs.Image(label='Content Image')
input2 = gr.inputs.Image(label='Style Image')

print("Launch...")
gr.Interface(
    fn=style_transfer,
    inputs=[input1, input2],
    outputs='image',
    title="Natural Style Transfer",
	server_name='0.0.0.0'
    ).launch()
