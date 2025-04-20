import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from mtcnn import MTCNN
import scipy.io as sio
from tensorflow.keras import layers
from collections import defaultdict 

DATASET_ROOT = r"path\to\300W_LP"

def collect_pairs(root_dir):
    pairs = []
    for subdir in ["AFW", "HELEN", "IBUG", "LFPW", 
                   "AFW_flip", "HELEN_flip", "IBUG_flip", "LFPW_flip"]:
        subdir_path = os.path.join(root_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        
        # Group files by base identifier (e.g., "AFW_134212_1")
        groups = defaultdict(list)
        for file in os.listdir(subdir_path):
            if file.endswith(".jpg"):
                base = "_".join(file.split("_")[:-1])
                groups[base].append(file)
        
        # Create pairs
        for base, files in groups.items():
            frontal = f"{base}_0.jpg"
            if frontal in files:
                for file in files:
                    if file != frontal:
                        profile_path = os.path.join(subdir_path, file)
                        frontal_path = os.path.join(subdir_path, frontal)
                        pairs.append((profile_path, frontal_path))
    return pairs

pairs = collect_pairs(DATASET_ROOT)
print(f"Total pairs: {len(pairs)}")

np.random.seed(42)
np.random.shuffle(pairs)
split_index = int(0.8*len(pairs))
train_pairs = pairs[:split_index]
test_pairs = pairs[split_index:]


def load_and_preprocess(profile_path, frontal_path):
    # Load images
    profile_img = tf.io.read_file(profile_path)
    profile_img = tf.image.decode_jpeg(profile_img, channels=3)
    frontal_img = tf.io.read_file(frontal_path)
    frontal_img = tf.image.decode_jpeg(frontal_img, channels=3)
    
    # Resize and normalize
    profile_img = tf.image.resize(profile_img, (128, 128))
    profile_img = (profile_img / 127.5) - 1.0  # [-1, 1]
    frontal_img = tf.image.resize(frontal_img, (128, 128))
    frontal_img = (frontal_img / 127.5) - 1.0
    return profile_img, frontal_img


def build_generator():
    inputs = layers.Input(shape=(128, 128, 3))
    
    # Encoder
    d1 = layers.Conv2D(64, 4, strides=2, padding='same')(inputs)
    d1 = layers.LeakyReLU(0.2)(d1)
    d2 = layers.Conv2D(128, 4, strides=2, padding='same')(d1)
    d2 = layers.BatchNormalization()(d2)
    d2 = layers.LeakyReLU(0.2)(d2)
    d3 = layers.Conv2D(256, 4, strides=2, padding='same')(d2)
    d3 = layers.BatchNormalization()(d3)
    d3 = layers.LeakyReLU(0.2)(d3)
    
    # Bottleneck
    bottleneck = layers.Conv2D(512, 4, strides=2, padding='same')(d3)
    bottleneck = layers.BatchNormalization()(bottleneck)
    bottleneck = layers.ReLU()(bottleneck)
    
    # Decoder
    u1 = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(bottleneck)
    u1 = layers.BatchNormalization()(u1)
    u1 = layers.ReLU()(u1)
    u1 = layers.Concatenate()([u1, d3])
    
    u2 = layers.Conv2DTranspose(128, 4, strides=2, padding='same')(u1)
    u2 = layers.BatchNormalization()(u2)
    u2 = layers.ReLU()(u2)
    u2 = layers.Concatenate()([u2, d2])
    
    u3 = layers.Conv2DTranspose(64, 4, strides=2, padding='same')(u2)
    u3 = layers.BatchNormalization()(u3)
    u3 = layers.ReLU()(u3)
    u3 = layers.Concatenate()([u3, d1])
    
    outputs = layers.Conv2DTranspose(3, 4, strides=2, padding='same', activation='tanh')(u3)
    return tf.keras.Model(inputs, outputs)



generator = build_generator()
discriminator = build_discriminator()

opt_g = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
opt_d = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

bce = tf.keras.losses.BinaryCrossentropy()
mae = tf.keras.losses.MeanAbsoluteError()


def train_step(profile_imgs, frontal_imgs):
    with tf.GradientTape(persistent=True) as tape:
        # Generate fake frontal images
        gen_imgs = generator(profile_imgs, training=True)
        
        # Discriminator loss
        real_output = discriminator(frontal_imgs, training=True)
        fake_output = discriminator(gen_imgs, training=True)
        loss_d_real = bce(tf.ones_like(real_output), real_output)
        loss_d_fake = bce(tf.zeros_like(fake_output), fake_output)
        loss_d = (loss_d_real + loss_d_fake) * 0.5
        
        # Generator losses
        loss_g_adv = bce(tf.ones_like(fake_output), fake_output)
        loss_pixel = mae(frontal_imgs, gen_imgs)
        loss_g = loss_g_adv + 100 * loss_pixel
    
    # Update discriminator
    grads_d = tape.gradient(loss_d, discriminator.trainable_variables)
    opt_d.apply_gradients(zip(grads_d, discriminator.trainable_variables))
    
    # Update generator
    grads_g = tape.gradient(loss_g, generator.trainable_variables)
    opt_g.apply_gradients(zip(grads_g, generator.trainable_variables))
    
    return loss_d, loss_g


EPOCHS = 10
for epoch in range(EPOCHS):
    for batch, (profile_batch, frontal_batch) in enumerate(train_dataset):
        loss_d, loss_g = train_step(profile_batch, frontal_batch)
        
        if batch % 50 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Batch {batch}, "
                  f"Loss D: {loss_d:.4f}, Loss G: {loss_g:.4f}")


def frontalize_image(image_path):
    # Load and preprocess
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))
    image = (image.astype(np.float32) / 127.5) - 1.0
    
    # Generate frontal view
    gen_image = generator.predict(image[np.newaxis, ...])[0] 
    gen_image = (gen_image * 0.5 + 0.5) * 255  # Denormalize
    return gen_image.astype(np.uint8)



test_image_path = r"C:\path\to\HELEN_10405424_1_8.jpg"
frontalized = frontalize_image(test_image_path)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(cv2.imread(test_image_path), cv2.COLOR_BGR2RGB))
plt.title("Input Profile")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(frontalized)
plt.title("Frontalized Output")
plt.axis('off')
plt.show()




discriminator.save_weights("C:path/to/weight/dis_weight.h5")  



from tensorflow.image import ssim,psnr
# Number of examples to plot
N = 3

# Grab one batch from your test_dataset
profile_batch, frontal_batch = next(iter(test_dataset))

# Generate frontal images
generated_batch = generator(profile_batch, training=False)

# Denormalize from [-1,1] → [0,1]
profile_batch = (profile_batch + 1.0) / 2.0
frontal_batch = (frontal_batch + 1.0) / 2.0
generated_batch = (generated_batch + 1.0) / 2.0

for i in range(min(N, profile_batch.shape[0])):
    prof = profile_batch[i].numpy()
    gen  = generated_batch[i].numpy()
    true = frontal_batch[i].numpy()
    
    # Compute metrics
    ssim_val = ssim(gen, true, max_val=1.0).numpy()
    psnr_val = psnr(gen, true, max_val=1.0).numpy()
    
    # Plot
    plt.figure(figsize=(12, 4))
    
    # 1) Input profile
    ax = plt.subplot(1, 3, 1)
    ax.imshow(prof)
    ax.set_title("Input Profile")
    ax.axis("off")
    
    # 2) Generated + metrics
    ax = plt.subplot(1, 3, 2)
    ax.imshow(gen)
    ax.set_title(f"Generated\nSSIM={ssim_val:.4f}\nPSNR={psnr_val:.1f} dB")
    ax.axis("off")
    
    # 3) Ground truth frontal
    ax = plt.subplot(1, 3, 3)
    ax.imshow(true)
    ax.set_title("Ground Truth")
    ax.axis("off")

    plt.show()

test_image_path = r"path\to\LFPW_image_test_0009_3.jpg"
frontalized = frontalize_image(test_image_path)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(cv2.imread(test_image_path), cv2.COLOR_BGR2RGB))
plt.title("Input Profile")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(frontalized)
plt.title("Frontalized Output")
plt.axis('off')
plt.show()
