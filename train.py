import tensorflow as tf
import numpy as np

#bool(input(f"Do you want to take ratio as input? [1/0]: "))

# define the diffusion model
class DiffusionModel(tf.keras.Model):
    def __init__(self):
        super(DiffusionModel, self).__init__()

        #self.take_img_ratio_as_input = take_img_ratio_as_input
        
        # convolutional layers for image processing
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')
        self.conv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')
        
        # LSTM layer for text processing
        self.lstm = tf.keras.layers.LSTM(units=64)
        
        # dense layers for combining image and text features
        self.dense1 = tf.keras.layers.Dense(units=256, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=32*32*3, activation='sigmoid')

        # reshape layer for after dense
        self.reshape = tf.keras.layers.Reshape((32,32,3))

        # flatten layer
        self.flatten = tf.keras.layers.Flatten()
        
        # convolutional layer for final output
        #self.conv4 = tf.keras.layers.Conv2D(filters=3, kernel_size=(3, 3), activation='sigmoid', padding='same')
        
    def call(self, inputs):
        # unpack inputs
        if len(inputs) == 3:
            image, text_embedding, ratio = inputs
        else:
            image, text_embedding = inputs
        
        # process image
        x = self.conv1(image)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # process text embedding
        y = self.lstm(text_embedding)

        #flatten
        x = self.flatten(x)
        y = self.flatten(y)
        if len(inputs) == 3:
            r = self.flatten(ratio)
        
        # combine image and text features
        if len(inputs) == 3:
            z = tf.concat([x, y, r], axis=-1)
        else:
            z = tf.concat([x, y], axis=-1)
        z = self.dense1(z)
        z = self.dense2(z)
        
        # generate final output
        output = self.reshape(z)
        #output = self.conv4(z)
        
        return output

# create an instance of the model
model = DiffusionModel()

# define the loss function
def loss_fn(image, reconstructed_image):
    image = tf.cast(image, tf.float32)
    reconstructed_image = tf.cast(reconstructed_image, tf.float32)
    mse = tf.reduce_mean(tf.square(image - reconstructed_image))
    return mse

# define the optimizer
optimizer = tf.keras.optimizers.Adam()

def add_noise(image, noise_level=0.075):
    """
    Returns a noisy image. 1 noise_level is 100% noise, 0 is 0% noise
    """
    noise = np.random.random(size=image.shape)
    probability = np.random.random(size=image.shape)
    mask = probability < noise_level
    new_image = np.copy(image)
    new_image[mask] = noise[mask]
    return new_image


# define the training step function
@tf.function
def batch_train_step(*inputs):
    '''
    images, degraded_images, text_embeddings
    '''
    images, degraded_images, text_embeddings, ratio = inputs
    
    if len(inputs) == 4:
        model_inputs = [degraded_images, text_embeddings, ratio]
    else:
        model_inputs = [degraded_images, text_embeddings]
    with tf.GradientTape() as tape:
        # forward pass
        reconstructed_images = model(model_inputs)
        # calculate loss
        loss = loss_fn(images, reconstructed_images)
        # calculate gradients
        gradients = tape.gradient(loss, model.trainable_variables)
    # update weights
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss

# define the training step function
@tf.function
def loaded_batch_train_step(*inputs):
    '''
    images, degraded_images, text_embeddings
    '''
    if len(inputs) == 4:
        images, degraded_images, text_embeddings, ratio = inputs
    else:
        images, degraded_images, text_embeddings = inputs
    with tf.GradientTape() as tape:
        # Get the model's signatures
        infer = model.signatures['serving_default']
        # forward pass
        if len(inputs) == 4:
            reconstructed_images = infer(input_1=degraded_images, input_2=text_embeddings, input_3=ratio)['output_1']
        else:
            reconstructed_images = infer(input_1=degraded_images, input_2=text_embeddings)['output_1']
        
        # calculate loss
        loss = loss_fn(images, reconstructed_images)
        # calculate gradients
        gradients = tape.gradient(loss, model.trainable_variables)
    # update weights
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss

# example usage

"""image = np.random.rand(32, 32, 3)
text_embedding = np.random.rand(768,1)
loss = train_step(np.array([image]), np.array([add_noise(image)]), np.array([text_embedding]))
print(loss)
model.summary()
"""