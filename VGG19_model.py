from keras.models import load_model
from flask import Flask,render_template,request
import cv2
from keras.applications.vgg19 import preprocess_input
import numpy as np
import base64

model=load_model('vgg19.h5')

app= Flask(__name__)

@app.route('/',methods=['GET'])
def index():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def main():
    image_file=request.files['image']
    img=image_file.read()
    img=np.frombuffer(img,np.uint8)
    img=cv2.imdecode(img,cv2.IMREAD_COLOR)
    img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    img=cv2.resize(img,(224,224))
    img=np.expand_dims(img,axis=0)
    img=preprocess_input(img)

    prediction=model.predict(img)
    pred_class=np.argmax(prediction)

    with open('imagenet_classes.txt','r') as f:
        classes=eval(f.read())
    pred_class=classes[pred_class]

    _,image_encoded=cv2.imencode('.png',img_rgb)
    img_base64=base64.b64encode(image_encoded).decode('utf-8')

    return render_template('predict.html',image_base64=img_base64,predicted_label=pred_class)



if __name__=='__main__':
    app.run(debug=True)