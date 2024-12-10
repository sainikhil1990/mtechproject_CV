from flask import Flask, request, redirect, url_for, render_template
from flask import Flask, render_template, jsonify, send_file

import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

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
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        return f'File successfully uploaded to {file_path}'

@app.route('/run-script', methods=['POST'])
def run_script():
    try:
        # Replace 'your_script.py' with the path to your Python script
        result = subprocess.run(['python3', 'object_detection.py'], capture_output=True, text=True)
        return jsonify({'output': result.stdout, 'error': result.stderr})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/get-image')
def get_image():
    image_path = '/home/eiiv-nn1-l3t04/Project/Algo/cropped_image.jpg'  # Path to your image
    return send_file(image_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)

