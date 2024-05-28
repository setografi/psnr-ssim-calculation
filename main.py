from flask import Flask, request, render_template
import numpy as np
import os
from PIL import Image
from utils import calculate_psnr, calculate_ssim

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'original_file' not in request.files or 'steganography_file' not in request.files:
        return 'No file part'
    
    original_file = request.files['original_file']
    steganography_file = request.files['steganography_file']
    
    if original_file.filename == '' or steganography_file.filename == '':
        return 'No selected file'
    
    if original_file and steganography_file:
        original_file_path = os.path.join(app.config['UPLOAD_FOLDER'], original_file.filename)
        steganography_file_path = os.path.join(app.config['UPLOAD_FOLDER'], steganography_file.filename)
        
        original_file.save(original_file_path)
        steganography_file.save(steganography_file_path)
        
        # Baca gambar sebagai numpy array
        original_image = read_image(original_file_path)
        steganography_image = read_image(steganography_file_path)
        
        # Hitung PSNR dan SSIM
        psnr = calculate_psnr(original_image, steganography_image)
        ssim = calculate_ssim(original_image, steganography_image)
        
        # Membulatkan hasil PSNR dan SSIM
        psnr = round(psnr, 3)
        ssim = round(ssim, 3)

        return render_template('result.html', psnr=psnr, ssim=ssim)

def read_image(file_path):
    # Menggunakan Pillow untuk membaca gambar
    img = Image.open(file_path)
    img = img.convert('RGB')  # Konversi gambar menjadi RGB
    return np.array(img)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
