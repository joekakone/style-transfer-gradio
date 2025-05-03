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


GRADIO_SERVER_NAME = "0.0.0.0"
GRADIO_SERVER_PORT = 7862
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
	print("Get Images....")
	content_image = preprocess_input(input_content_image)
	style_image = preprocess_input(input_style_image)

	try:
		print("Apply Style....")
		stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]
		stylized_image = np.array(stylized_image)
		stylized_image = np.reshape(stylized_image, stylized_image.shape[1:])
	except Exception as e:
		print("Error: ", e)
		stylized_image = np.array(content_image)
		stylized_image = np.reshape(stylized_image, stylized_image.shape[1:])

	print("Style Applied....")
	print(type(stylized_image))

	return stylized_image


def main():
	# retreive model from tfhub
	print("Download model from Tensorflow Hub...")
	global hub_module # Make variable available in functions
	hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

	# inputs
	input1 = gr.Image(label='Content Image')
	input2 = gr.Image(label='Style Image')

	print("Launch...")
	gr.Interface(
		fn=style_transfer,
		inputs=[input1, input2],
		outputs='image',
		title="Natural Style Transfer",
		).launch(
			server_name=GRADIO_SERVER_NAME,
			server_port=GRADIO_SERVER_PORT
	)


if __name__ == '__main__':
  main()

