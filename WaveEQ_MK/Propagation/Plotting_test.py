import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
from scipy.fft import fftfreq, fftshift

# Function to create the GIF
def create_gif(data, gif_filename,dt,w, fps=10):
    num_frames = data.shape[1]
    images = []
    t_full = np.arange(data.shape[0]) * dt
    
    for i in range(num_frames):
        plt.figure(figsize=(6, 4))
        plt.plot(t_full/(2 * np.pi / w),data[:, i], label=f'Slice {i}')
        plt.xlabel('Time (Cycles of electric field)')
        plt.ylabel('Intensity')
        plt.title(f'Slice {i}')
        plt.legend()
        plt.grid(True)
        
        # Save plot to a BytesIO object
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        
        # Read the image from the BytesIO object
        buf.seek(0)
        img = Image.open(buf)
        images.append(img)
    
    # Save all images as a GIF
    images[0].save(gif_filename, save_all=True, append_images=images[1:], loop=0, duration=1000//fps)

# # Example usage
# if #__name__ == "__main__":
#     # Create a random array for demonstration
#     # Replace this with loading your actual data array
#     #data = #Replace with actual data 
    
#     # Create GIF
#     create_gif(data, 'output.gif',dt,w, fps=10)


# Function to create the GIF
def create_gif_freq(data, gif_filename,w, dt, fps=10):
    num_frames = data.shape[1]
    images = []
    
    # Define the frequency axis
    t_full = np.arange(data.shape[0]) * dt
    freq_axis = (2 * np.pi) * fftshift(fftfreq(len(t_full), d=dt))
    
    for i in range(num_frames):
        plt.figure(figsize=(10, 6))
        plt.semilogy(freq_axis/w, data[:, i], label=f'Slice{i}')
        plt.xlabel('Harmonic Order')
        plt.ylabel('Intensity')
        plt.title(f'Slice {i}')
        plt.xlim(-10,100)
        plt.legend()
        plt.grid(True)
        
        # Save plot to a BytesIO object
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        
        # Read the image from the BytesIO object
        buf.seek(0)
        img = Image.open(buf)
        images.append(img)
    
    # Save all images as a GIF
    images[0].save(gif_filename, save_all=True, append_images=images[1:], loop=0, duration=1000//fps)

# # Example usage
# if __name__ == "__main__":
#     # Parameters
#     dt = 0.1  # Example time step, adjust based on your data
    
#     # Create a random array for demonstration
#     # Replace this with loading your actual data array
#     data = np.random.rand(4409, 200)
    
#     # Create GIF
#     create_gif_freq(data, 'output_frequency.gif',w, dt, fps=10)