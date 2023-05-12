print("importing...")
import tensorflow as tf
import numpy as np
import sentence_similarity as ss
import matplotlib.pyplot as plt
import os

display_during_inference = True
ratio = 1

ratio = np.array([ratio])
ratio = tf.constant(ratio, np.float32)

# Load the SavedModel
print("loading model...")

model_path = "latest" # 'models/row_22640-Y2023M5D9H22M21S27'
if model_path == "latest":
    model_path = os.listdir("models")[-1]

print(f"loading model from models/{model_path}")

model = tf.saved_model.load("models/"+model_path)

# Get the model's signatures
infer = model.signatures['serving_default']

# Prepare input data
print("preparing input data")
input_image = np.array([np.random.random((32, 32, 3))])
input_text = input("Enter the prompt: ")
input_embeddings = ss.get_bert_embedding(input_text).reshape((1,-1,1))
print(f"image shape: {input_image.shape}\nembedding shape: {input_embeddings.shape}\nratio shape: {ratio.shape}")
step_num = int(input("Enter number of steps to run: "))

# Run inference
print("running inference")
for i in range(step_num):
    print(f"doing step {i+1}/{step_num}",end=" ")
    #output = model([input_image, input_embeddings], training=False)
    output = infer(input_1 = tf.constant(input_image, np.float32), input_2 = tf.constant(input_embeddings, np.float32), input_3=ratio)
    print(np.max(output['output_1'][0]), np.min(output['output_1'][0]),end="\r")
    if display_during_inference:
        plt.imshow(output['output_1'][0])
        plt.ion()
        plt.show()
        plt.pause(0.1)
    input_image = np.array(output['output_1'])

# Post-process the output
result = output['output_1']
plt.imshow(result[0])
plt.show()