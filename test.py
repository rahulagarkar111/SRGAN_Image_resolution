import cv2
from PIL import Image
from keras.models import load_model
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import random

generator = load_model('gen_e26_3.h5', compile=False)

def preprocess_image(image):
    # Convert uploaded image to numpy array and resize to (32, 32)
    image = cv2.resize(image, (32, 32))
    # Normalize pixel values to range [0, 1]
    # image = image / 255.0
    return image

def main():
    st.title("Image Resolution using SRGAN")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Save the image as JPG
        if st.button('Save as JPG'):
            try:
                save_path = 'uploaded_image.jpg'
                image.save(save_path, 'JPEG')
                st.success(f"Image saved as {save_path}")
                image.close()  # Close the image file
            except Exception as e:
                st.error(f"Error saving image: {e}")

      
        img_lr = cv2.imread("uploaded_image.jpg")
        ori_img= cv2.imread("uploaded_image.jpg")
        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        
        img_lr = cv2.resize(img_lr, (32,32), interpolation=cv2.INTER_LANCZOS4)
        img_lr = cv2.cvtColor(img_lr, cv2.COLOR_BGR2RGB)
        # img_lr = np.array(img_lr)

        img_hr = cv2.imread("uploaded_image.jpg")

        img_hr = cv2.resize(img_hr, (128, 128), interpolation=cv2.INTER_LANCZOS4)
        img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2RGB)
        # img_hr = np.array(img_hr)

        img_lr = img_lr / 255.0
        img_hr = img_hr / 255.0
        ori_img = ori_img / 255.0

        img_lr = np.expand_dims(img_lr, axis=0)
        img_hr = np.expand_dims(img_hr, axis=0)
        ori_img= np.expand_dims(ori_img, axis=0)

        generated__img = generator.predict(img_lr)
        generated__img = np.clip(generated__img, 0.0, 1.0)

       
        fig, axs = plt.subplots(1, 3, figsize=(18,6))

        # Display the LR Image in the first subplot
        axs[0].set_title('LR Image')
        axs[0].imshow(img_lr[0,:,:,:])
        axs[0].axis('off')

        axs[1].set_title('Superresolution')
        axs[1].imshow(img_hr[0,:,:,:])
        axs[1].axis('off')

        
        # Display the Orig. HR Image in the third subplot
      
        axs[2].set_title('HR Image')
        axs[2].imshow(ori_img[0,:,:,:], aspect='auto')
        axs[2].axis('off')


        # Adjust layout
        plt.tight_layout()

        # Render the Matplotlib figure using Streamlit
        st.pyplot(fig)




    

if __name__ == "__main__":
    main()
