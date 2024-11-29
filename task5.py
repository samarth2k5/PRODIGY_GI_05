import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Load an image and convert it into a tensor
def load_img(path_to_img):
    img = Image.open(path_to_img)
    img = img.resize((512, 512))  # Resize image for faster processing
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    return tf.convert_to_tensor(img, dtype=tf.float32)

# Preprocess image for neural network input
def preprocess_img(img):
    return img / 255.0

# De-process image to a form that can be displayed
def deprocess_img(img):
    img = img.numpy()
    img = img.squeeze()
    img = np.clip(img * 255, 0, 255).astype('uint8')
    return img

# Display images
def imshow(img, title=""):
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Load pre-trained VGG19 model for feature extraction
def load_vgg_model():
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    return vgg

# Extract features from a layer of the VGG model
def get_features(model, img, layers):
    features = []
    for layer_name in layers:
        layer = model.get_layer(layer_name)
        feature = layer(img)
        features.append(feature)
    return features

# Content and style loss functions
def content_loss(content, target):
    return tf.reduce_mean(tf.square(content - target))

def gram_matrix(input_tensor):
    # Compute the Gram matrix for style representation
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    return result / tf.cast(input_tensor.shape[1] * input_tensor.shape[2], tf.float32)

def style_loss(style, target):
    s_gram = gram_matrix(style)
    t_gram = gram_matrix(target)
    return tf.reduce_mean(tf.square(s_gram - t_gram))

def total_variation_loss(image):
    return tf.reduce_sum(tf.image.total_variation(image))

# Function to compute total loss
def compute_loss(content_image, style_image, generated_image, content_weight=1e3, style_weight=1e-2, tv_weight=1e-6):
    content_layers = ['block5_conv2']
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    
    vgg_model = load_vgg_model()
    
    # Get features of the content, style, and generated images
    content_features = get_features(vgg_model, content_image, content_layers)
    style_features = get_features(vgg_model, style_image, style_layers)
    generated_features = get_features(vgg_model, generated_image, content_layers + style_layers)
    
    # Calculate losses
    c_loss = content_loss(content_features[0], generated_features[0])
    
    s_loss = 0
    for sf, gf in zip(style_features, generated_features[1:]):
        s_loss += style_loss(sf, gf)
    
    tv_loss = total_variation_loss(generated_image)
    
    total_loss = content_weight * c_loss + style_weight * s_loss + tv_weight * tv_loss
    return total_loss

# Optimizer setup
optimizer = tf.optimizers.Adam(learning_rate=5.0)

# Function to run the style transfer
def run_style_transfer(content_path, style_path, num_iterations=1000):
    content_image = load_img(content_path)
    style_image = load_img(style_path)
    
    # Preprocess images
    content_image = preprocess_img(content_image)
    style_image = preprocess_img(style_image)
    
    # Create a copy of the content image to generate new image
    generated_image = tf.Variable(content_image, dtype=tf.float32)
    
    for i in range(num_iterations):
        with tf.GradientTape() as tape:
            # Compute the total loss for the current iteration
            loss = compute_loss(content_image, style_image, generated_image)
        
        # Get gradients with respect to the generated image
        gradients = tape.gradient(loss, generated_image)
        
        # Apply the gradients to update the generated image
        optimizer.apply_gradients([(gradients, generated_image)])
        
        # Print the loss every 100 iterations
        if i % 100 == 0:
            print(f"Iteration {i}: Loss = {loss.numpy()}")
        
        # Display the result at regular intervals
        if i % 100 == 0:
            imshow(deprocess_img(generated_image), title=f"Iteration {i}")
    
    return generated_image

# Run the style transfer
content_image_path = 'path_to_your_content_image.jpg'  # Replace with your content image path
style_image_path = 'path_to_your_style_image.jpg'      # Replace with your style image path

generated_image = run_style_transfer(content_image_path, style_image_path)
final_image = deprocess_img(generated_image)
imshow(final_image, title="Final Generated Image")
