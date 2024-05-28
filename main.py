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
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        # Simpan file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Baca gambar sebagai numpy array
        image = read_image(file_path)

        # Proses embedding teks ke dalam gambar (dummy function)
        encrypted_image = embed_text_into_image(image, "Your text here")

        # Hitung PSNR dan SSIM
        psnr_value = calculate_psnr(image, encrypted_image)
        ssim_value = calculate_ssim(image, encrypted_image)

        return render_template('result.html', psnr=psnr_value, ssim=ssim_value)

def read_image(file_path):
    # Menggunakan Pillow untuk membaca gambar
    img = Image.open(file_path)
    img = img.convert('RGB')  # Konversi gambar menjadi RGB
    return np.array(img)

def embed_text_into_image(image, text):
    # Implementasi dummy, ganti dengan fungsi embedding yang sebenarnya
    return image

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
