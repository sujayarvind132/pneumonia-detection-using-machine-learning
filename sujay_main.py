import tkinter as tk
import tkinter.filedialog as filedialog
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from PIL import ImageTk, Image, ImageChops
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load the pre-trained model
model = keras.models.load_model('chest_xray.h5')

# Create a GUI window
window = tk.Tk()
window.title("Radiologist Assistant Using Machine Learning")

# Create a function to handle the "Upload" button click event
def is_bw_image(file_path):
    # Open the image file
    with Image.open(file_path) as img:
        # Convert the image to grayscale and then back to RGB
        # If the image is already grayscale, this won't change anything
        rgb_img = img.convert('RGB')
        gray_img = rgb_img.convert('L')
        rgb_gray_img = gray_img.convert('RGB')

        # Calculate the root-mean-square (RMS) difference between the original
        # RGB image and the grayscale RGB image
        rms = rms_diff(rgb_img, rgb_gray_img)

        # If the RMS difference is less than 1, the image is considered black and white
        return rms < 1

# Calculate the root-mean-square (RMS) difference between two images
def rms_diff(im1, im2):
    diff = ImageChops.difference(im1, im2)
    h = diff.histogram()
    sq = (val * ((idx % 256) ** 2) for idx, val in enumerate(h))
    sum_sqs = sum(sq)
    return round(pow(sum_sqs / float(im1.size[0] * im1.size[1]), 0.5), 2)

def upload_image():
    # Use file dialog to select an image file
    file_path = filedialog.askopenfilename()
    if not is_bw_image(file_path):
        result_label.configure(text="WARNING!\nPLEASE GIVE VALID INPUT", font=("Arial", 30, "bold"), fg="red")
        return
    # Load the selected image
    img = keras.preprocessing.image.load_img(file_path, target_size=(224, 224))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.

    # Display the selected image in the GUI window
    selected_image = ImageTk.PhotoImage(Image.open(file_path).resize((300, 300)))
    selected_image_label.configure(image=selected_image)
    selected_image_label.image = selected_image

    # Save the image array for later use
    global selected_img_array, selected_img_name
    selected_img_array = img_array
    selected_img_name = file_path.split("/")[-1]

    # Enable the "Predict" button
    predict_button.config(state="normal")

    # Clear the result label and output plot
    result_label.configure(text="")
    output_plot.clear()
    output_plot.axis("off")
    output_plot.set_title("Result")

    # Clear the result dot
    result_dot.delete("all")

# Create a function to handle the "Predict" button click event
result_dot = tk.Canvas(window, width=20, height=20, bg="white")
result_dot.pack()

def predict_image():
    global selected_img_array, selected_img_name

    # Make a prediction
    prediction = model.predict(selected_img_array)[0]

    # Display the prediction result in the GUI window
    #percentage_pneumonia = int(prediction[1] * 100)
    if prediction[0] < prediction[1]:
        result_label.configure(text=f"Prediction for {selected_img_name}:  Pneumonia is Present",
                               font=("Arial", 18, "bold"), fg="red")
        dot_color = "red"
    else:
        result_label.configure(text=f"Prediction for {selected_img_name}: Pneumonia is Absent", font=("Arial", 18, "bold"),
                               fg="green")
        dot_color = "green"

    # Create a canvas to draw the dot
    result_dot_canvas.delete("all")
    result_dot_canvas.config(width=30, height=30)
    result_dot_canvas.pack()
    result_dot_canvas.create_oval(5, 5, 25, 25, fill=dot_color, outline=dot_color)

    output_plot.imshow(selected_img_array[0], cmap='gray')
    output_plot.axis("off")
    output_canvas.draw()
# Create a label for the heading
heading_label = tk.Label(window, text="Radiologist Assistant Using Machine Learning \n Pneumonia Detection", font=("Arial", 24, "bold"))
heading_label.pack()

# Create an "Upload" button
upload_button = tk.Button(window, text="Upload", font=("Arial", 16), command=upload_image)
upload_button.pack()

# Create a label to display the selected image
selected_image_label = tk.Label(window)
selected_image_label.pack()

# Create a "Predict" button
predict_button = tk.Button(window, text="Predict", font=("Arial", 16), command=predict_image, state="disabled")
predict_button.pack()

result_label = tk.Label(window, font=("Arial", 18, "bold"))
result_label.pack()

result_dot_canvas = tk.Canvas(window, width=20, height=20)
result_dot_canvas.pack()

output_figure = plt.figure(figsize=(5, 5), dpi=100)
output_plot = output_figure.add_subplot(111)
output_plot.axis("off")
output_plot.set_title("Result")

output_canvas = FigureCanvasTkAgg(output_figure, master=window)
output_canvas.draw()
output_canvas.get_tk_widget().pack()

window.mainloop()

