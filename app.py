from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.models import load_model
import pickle
import numpy as np
from PIL import Image


def extract_features(filename, model):
        try:
            image = Image.open(filename)
            
        except:
            print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
        image = image.resize((299,299))
        image = np.array(image)
        # for images that has 4 channels, we convert them into 3 channels
        if image.shape[2] == 4: 
            image = image[..., :3]
        image = np.expand_dims(image, axis=0)
        image = image/127.5
        image = image - 1.0
        feature = model.predict(image)
        return feature

def word_for_id(integer, tokenizer):
 for word, index in tokenizer.word_index.items():
     if index == integer:
         return word
 return None


def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    print(in_text)
    return in_text

# Recreate and Save the Tokenizer
texts = []  # Add your training texts here
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

with open('tokenizer.pkl', 'wb') as tokenizer_file:
    pickle.dump(tokenizer, tokenizer_file)

max_length = 32
model = load_model("model_9.h5")

xception_model = Xception(include_top=False, pooling="avg") 

# img_path = "C:/Users/rajni/OneDrive/Desktop/piyush jain/image_captioning/piyush/static/uploads/hill_climbing.jpg"
# photo = extract_features(img_path, xception_model)
# description = generate_desc(model, tokenizer, photo, max_length)
# print("\n\n")
# print(description)


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_image', filename=filename))

    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_image(filename):
    
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    photo = extract_features(img_path, xception_model)
    
    # Load the tokenizer
    with open("tokenizer.pkl", "rb") as tokenizer_file:
        tokenizer = pickle.load(tokenizer_file)
        
    
    description = generate_desc(model, tokenizer, photo, max_length)
    print(description)
    print("HI desc 1")
    #description = description[6:-4]
    print("HI desc 2")
    print(description)
    return render_template('uploaded.html', filename=filename, description=description)
    

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(debug=True)