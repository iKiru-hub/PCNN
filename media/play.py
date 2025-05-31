from tkinter import Tk, Label
from PIL import Image, ImageTk

# Create a Tkinter window
root = Tk()
root.title("GIF Player")

# Load the GIF
gif_path = "spav2_383.gif"  # Replace with your GIF path
gif = Image.open(gif_path)

# Set the window size to match the GIF dimensions
root.geometry(f"{int(gif.width)}x{int(gif.height)}")

# Create a Label to display the GIF
label = Label(root)
label.pack()

# Function to update frames
def update_frame(frame_index):
    try:
        gif.seek(frame_index)  # Go to the next frame
        frame = ImageTk.PhotoImage(gif)
        label.config(image=frame)
        label.image = frame
        root.after(100, update_frame, frame_index + 1)  # Adjust delay as needed
    except EOFError:
        gif.seek(0)  # Restart the GIF
        update_frame(0)

# Start playing the GIF
update_frame(0)

# Run the Tkinter event loop
root.mainloop()
