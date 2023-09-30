import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import rembg

def welcome_page():
    st.title("Welcome to VisualAlchemy")
    
    st.markdown(
        """
        VisualAlchemy is a powerful image editing web app that allows you to transform and edit your images with ease. 
        You can convert image formats, apply filters, resize images, colorize black and white images, remove backgrounds,
        adjust brightness, rotate images, and much more.
        
        To get started, simply click the button below to proceed to the editing options.
        """
    )



def main():
    st.title("VisualAlchemy: Transform and Edit")

    st.sidebar.header("Edit Options")
    edit_option = st.sidebar.radio("Select an editing option:", ["Black and White", "Resize", "Colorize", "Background Removal", "Image editing"])

    image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
    if image_file is not None:
        image = Image.open(image_file)
        img = np.array(image)

        st.text("Your original image")
        st.image(image, use_column_width=True)

        if edit_option == "Black and White":
            bw_image = convert_to_black_and_white(image)
            st.image(bw_image, caption="Black and White", use_column_width=True)

            # Always download as JPEG
            download_image(bw_image, "edited_image.jpg")

        elif edit_option == "Resize":
            width = st.slider("New Width", 1, 1000, image.width)
            height = st.slider("New Height", 1, 1000, image.height)
            resized_image = resize_image(image, width, height)
            st.image(resized_image, caption=f"Resized to {width}x{height}", use_column_width=True)

            # Provide a download option for the resized image
            if st.button("Download Resized Image"):
                download_image(np.array(resized_image), "resized_image.jpg")


        elif edit_option == "Colorize":
            # Colorize black and white image
            colorized_image = colorize_image(img)
            st.image(colorized_image, caption="Colorized", use_column_width=True)

            # Provide a download option for the colorized image
            if st.button("Download Colorized Image"):
                download_image(colorized_image, "colorized_image.jpg")

        elif edit_option == "Background Removal":
            # Remove the image background using rembg library
            background_removed_image = remove_background(image)
            st.image(background_removed_image, caption="Background Removed", use_column_width=True)

            # Provide a download option for the background removed image
            if st.button("Download Background Removed Image"):
                download_background_removed_image(background_removed_image, "background_removed_image.png")

        elif edit_option == "Image editing":
            filter_option = st.selectbox("Select a filter:", ["Blur", "Edge Detection", "Grayscale", "Brightness", "Rotation"])
            if filter_option == "Blur":
                kernel_size = st.slider("Select kernel size:", 1, 10, step=2)  # Ensure odd kernel size
                filtered_image = apply_blur(img, kernel_size)
            elif filter_option == "Edge Detection":
                filtered_image = apply_edge_detection(img)
            elif filter_option == "Grayscale":
                filtered_image = apply_grayscale(img)
            elif filter_option == "Brightness":
                brightness_level = st.slider("Brightness Level:", -100, 100, 0)
                filtered_image = adjust_brightness(img, brightness_level)
            elif filter_option == "Rotation":
                rotation_angle = st.slider("Rotation Angle:", -180, 180, 0)
                filtered_image = rotate_image(img, rotation_angle)

            st.image(filtered_image, caption=f"{filter_option} Filtered", use_column_width=True)

            # Provide a download option for the filtered image
            if st.button("Download edited Image"):
                download_image(filtered_image, f"filtered_image_{filter_option.lower()}.jpg")


def read_image(image_file):
    image = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    return cv2.imdecode(image, cv2.IMREAD_COLOR)

def convert_format(image, new_format):
    # Convert image format using OpenCV
    _, encoded_image = cv2.imencode(f".{new_format.lower()}", image)
    return encoded_image.tobytes()

def convert_to_black_and_white(image):
    img = np.array(image)
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, bw_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(bw_image, cv2.COLOR_GRAY2RGB)

def resize_image(image, width, height):
    # Resize the PIL image
    resized_image = image.resize((width, height))

    return resized_image

def colorize_image(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

    net = cv2.dnn.readNetFromCaffe('colorization_deploy_v2.prototxt','colorization_release_v2.caffemodel')
    pts = np.load('pts_in_hull.npy')

    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)

    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    scaled = img_gray.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)

    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (img_gray.shape[1], img_gray.shape[0]))

    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
    colorized = np.clip(colorized, 0, 1)
  
    colorized = (255 * colorized).astype("uint8")
    return colorized

def remove_background(image):
    # Convert the image to bytes
    img_byte_array = BytesIO()
    image.save(img_byte_array, format='PNG')
    img_byte_array = img_byte_array.getvalue()

    # Use the rembg library to remove background
    result = rembg.remove(img_byte_array)
    background_removed_image = Image.open(BytesIO(result)).convert("RGBA")

    return background_removed_image

def download_background_removed_image(image, filename):
    # Create a BytesIO object to hold the image data
    buf = BytesIO()
    # Save the background removed image to the BytesIO object in PNG format
    image.save(buf, format="PNG")
    buf.seek(0)

    # Provide the download link for the background removed image
    st.download_button(label="Download Background Removed Image", data=buf, file_name=filename)

def apply_blur(image, kernel_size):
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure kernel_size is odd
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def apply_edge_detection(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 100, 200)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def apply_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def adjust_brightness(image, level):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, level)
    v = np.clip(v, 0, 255)
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    # Create a BytesIO object to hold the image data
    buf = BytesIO()
    # Save the image to the BytesIO object in the specified format
    pil_image.save(buf, format=format)
    buf.seek(0)

    # Provide the download link
    st.download_button(label="Download Image", data=buf, file_name=filename)

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h))

def download_image(image, filename, format="JPEG"):
    # Convert the NumPy array to a PIL Image
    pil_image = Image.fromarray(image)

    # Create a BytesIO object to hold the image data
    buf = BytesIO()
    # Save the image to the BytesIO object in the specified format
    pil_image.save(buf, format=format.upper())  # Ensure format is in uppercase
    buf.seek(0)

    # Provide the download link
    st.download_button(label="Download Image", data=buf, file_name=filename)



if __name__ == "__main__":
    welcome_page()
    main()