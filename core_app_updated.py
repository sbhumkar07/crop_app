
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import keras.backend as K
from datetime import datetime as dt
import numpy as np
import cv2
from cv2 import resize, INTER_AREA
import uuid
from PIL import Image
import os
import tempfile
from keras.models import load_model
import imageio
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
import keras
from keras.applications import vgg16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions

"""
def resize_image_oct(image):
    resized_image = cv2.resize(image, (128,128)) #Resize all the images to 128X128 dimensions
    if(len(resized_image.shape)!=3):
        resized_image = cv2.cvtColor(resized_image,cv2.COLOR_GRAY2RGB) #Convert to RGB
    return resized_image
"""

"""
def resize_image_pnm(image):
    resized_image = cv2.resize(image, (128,128)) #Resize all the images to 128X128 dimensions
    if(len(resized_image.shape)!=3):
        resized_image = cv2.cvtColor(resized_image,cv2.COLOR_GRAY2RGB) #Convert to RGB
    return resized_image
"""
#Download VGG16 Weights.
#wget https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5
"""
def load_vgg16_model():
  input_shape = (224, 224, 3)

  #Instantiate an empty model
  vgg_model = Sequential([
  Conv2D(64, (3, 3), input_shape=input_shape, padding='same', activation='relu'),
  Conv2D(64, (3, 3), activation='relu', padding='same'),
  MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
  Conv2D(128, (3, 3), activation='relu', padding='same'),
  Conv2D(128, (3, 3), activation='relu', padding='same',),
  MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
  Conv2D(256, (3, 3), activation='relu', padding='same',),
  Conv2D(256, (3, 3), activation='relu', padding='same',),
  Conv2D(256, (3, 3), activation='relu', padding='same',),
  MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
  Conv2D(512, (3, 3), activation='relu', padding='same',),
  Conv2D(512, (3, 3), activation='relu', padding='same',),
  Conv2D(512, (3, 3), activation='relu', padding='same',),
  MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
  Conv2D(512, (3, 3), activation='relu', padding='same',),
  Conv2D(512, (3, 3), activation='relu', padding='same',),
  Conv2D(512, (3, 3), activation='relu', padding='same',),
  MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
  Flatten(),
  Dense(4096, activation='relu'),
  Dense(4096, activation='relu'),
  Dense(1000, activation='softmax')
  ])

  vgg_model.load_weights("static/weights/vgg16.h5")
  return vgg_model
"""


"""Instantiating the flask object"""
app = Flask(__name__)
CORS(app)

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/checkup')
def checkup():
    return render_template('checkup.html')

@app.route('/grape.html')
def brain():
    return render_template('grape.html')

@app.route('/tomato.html')
def malaria():
    return render_template('tomato.html')

"""
@app.route('/cancer.html')
def cancer():
    return render_template('cancer.html')

@app.route('/fun.html')
def fun():
    return render_template('fun.html')

@app.route('/oct.html')
def oct():
    return render_template('oct.html')

@app.route('/pnm.html')
def pnm():
    return render_template('pnm.html')

@app.route('/retino.html')
def retino():
    return render_template('retino.html')
"""
@app.route('/index.html')
def index_from_checkup():
    return render_template('index.html')

@app.route('/checkup.html')
def checkup_from_any():
    return render_template('checkup.html')

"""
@app.route('/blog.html')
def blog():
    return render_template('blog.html')

@app.route('/about.html')
def about():
    return render_template('about.html')
"""


@app.route("/", methods = ["POST", "GET"])
def index():
  if request.method == "POST":
    type_ = request.form.get("type", None)
    data = None
    final_json = []
    if 'img' in request.files:
      file_ = request.files['img']
      name = os.path.join(tempfile.gettempdir(), str(uuid.uuid4().hex[:10]))
      file_.save(name)
      print("[DEBUG: %s]"%dt.now(),name)

      if(type_=='tom' or type_=='grape'):
        test_image = image.load_img(name, target_size = (128, 128))
        test_image = image.img_to_array(test_image)
        test_image = test_image/255
        test_image = np.expand_dims(test_image, axis = 0)
        data=test_image

      elif(type_=='oct'):
        test_image = imageio.imread(name)                  #Read image using the PIL library
        test_image = resize_image_oct(test_image)          #Resize the images to 128x128 pixels
        test_image = np.array(test_image)                  #Convert the image to numpy array
        test_image = test_image/255                        #Scale the pixels between 0 and 1
        test_image = np.expand_dims(test_image, axis=0)    #Add another dimension because the model was trained on (n,128,128,3)
        data = test_image



     # model=get_model(type_)[0]

      if(type_=='tom'):
         model=load_model("static/weights/tomato.h5")
         pred_val = translate_tomato(model.predict(data))
         final_json.append({"empty": False, "type":type_,
                            "pred_val": pred_val})


      elif(type_=='grape'):
           model=load_model("static/weights/grape.h5")
           pred_val = translate_grape(model.predict(data))
           final_json.append({"empty": False, "type":type_,
                              "pred_val": pred_val})



    else:
      warn = "Feeding blank image won't work. Please enter an input image to continue."
      pred_val =" "
      final_json.append({"pred_val": warn,"para": " ","unin": " ","tumor": " ", "can":" ",
                         "normal": " ","bac": " ","viral": " ","cnv": " ","dme": " ",
                         "drusen": " ","mild": " ","mod": " ","severe": " ","norm": " ",
                         "top1": " ","top2": " ","top3": " ","top4": " ","top5": " "})

    K.clear_session()
    return jsonify(final_json)
  return jsonify({"empty":True})

"""This function is used to load the model from disk.
def load_model_(model_name):
  model_name = os.path.join("static/weights",model_name)
  model = load_model(model_name)
  return model
"""

"""This function is used to load the specific model for specific request calls. This
function will return a list of dictionary items, where the key will contain the loaded
models and the value will contain the request type.
def get_model(name = None):
  model_name = []
  if(name=='mal'):
    model_name.append({"model": load_model_("malaria.h5"), "type": name})
  elif(name=='brain'):
    model_name.append({"model": load_model_("brain_tumor.h5"), "type": name})
  elif(name=='pnm'):
    model_name.append({"model": load_model_("pneumonia.h5"), "type": name})
  elif(name=='oct'):
    model_name.append({"model": load_model_("retina_OCT.h5"), "type": name})
  elif(name=='dia_ret'):
    model_name.append({"model": load_model_("diabetes_retinopathy.h5"), "type": name})
  elif(name=='breast'):
    model_name.append({"model": load_model_("breastcancer.h5"), "type": name})
  elif(name=='fun'):
    model_name.append({"model": load_vgg16_model(), "type": name})
  return model_name
"""

def translate_tomato(preds):
  dicti=["Bacterial Spot - can do is ...","Early Blight","Late Blight","Leaf Mold","Septoria Leaf Spot","Spider_mites Two-spotted_spider_mite","Target Spot  - we can do is ...","Tomato_Yellow_Leaf_Curl_Virus","Tomato mosaic virus","Healthy"]
  return dicti[np.argmax(preds)]


def translate_grape(preds):
  dicti=["Grape___Black_rot","Grape___Esca_(Black_Measles)","Grape___Leaf_blight_(Isariopsis_Leaf_Spot)","Grape___healthy"]
  return dicti[np.argmax(preds)]

"""preds will contain the predictions made by the model. We will take the class probabalities and
store them in individual variables. We will return the class probabilities and the final predictions
made by the model to the frontend. The value contained in variables total and prediction will be
displayed in the frontend HTML layout."""
def translate_malaria(preds):
  y_proba_Class0 = preds.flatten().tolist()[0] * 100
  y_proba_Class1 = 100.0-y_proba_Class0

  para_prob="Probability of the cell image to be Parasitized: {:.2f}%".format(y_proba_Class1)
  unifected_prob="Probability of the cell image to be Uninfected: {:.2f}%".format(y_proba_Class0)

  total = para_prob + " " + unifected_prob
  total = [para_prob,unifected_prob]

  if (y_proba_Class1 > y_proba_Class0):
      prediction="Inference: The cell image shows strong evidence of Malaria."
      return total,prediction
  else:
      prediction="Inference: The cell image shows no evidence of Malaria."
      return total,prediction

"""This function also does the same thing as above. Since it's a two class classification problem,
we can subtract one probability percentage values from 100 to get the other value."""
def translate_cancer(preds):
  y_proba_Class0 = preds.flatten().tolist()[0] * 100
  y_proba_Class1 = 100.0-y_proba_Class0

  can="Probability of the histopathology image to have cancer: {:.2f}%".format(y_proba_Class1)
  norm="Probability of the histopathology image to be normal: {:.2f}%".format(y_proba_Class0)

  total = [can,norm]

  if (y_proba_Class1 > y_proba_Class0):
      prediction="Inference: The histopathology image shows strong evidence of Invasive Ductal Carcinoma."
      return total,prediction
  else:
      prediction="Inference: The cell image shows no evidence of Invasive Ductal Carcinoma."
      return total,prediction



if __name__=="__main__":
  app.run("0.0.0.0",80, debug = False)

"""Tis function will compute the values for the brain cancer model"""
"""
def translate_brain(preds):
  y_proba_Class0 = preds.flatten().tolist()[0] * 100
  y_proba_Class1 = 100.0-y_proba_Class0

  tumor="Probability of the MRI scan to have a brain tumor: {:.2f}%".format(y_proba_Class1)
  normal="Probability of the MRI scan to not have a brain tumor: {:.2f}%".format(y_proba_Class0)

  total = [tumor, normal]

  if (y_proba_Class1 > y_proba_Class0):
      prediction="Inference: The MRI scan has a brain tumor."
      return total,prediction
  else:
      prediction="Inference: The MRI scan does not show evidence of any brain tumor."
      return total,prediction
"""

"""For multi class problems, we will obtain each of the class probabilities for each of the
classes. We will send this values to frontend using a jsonfy object. The final jsonfy object will
contain """

"""
def translate_oct(preds):
  y_proba_Class0 = preds.flatten().tolist()[0] * 100
  y_proba_Class1 = preds.flatten().tolist()[1] * 100
  y_proba_Class2 = preds.flatten().tolist()[2] * 100
  y_proba_Class3 = preds.flatten().tolist()[3] * 100

  cnv="Probability of the input image to have Choroidal Neo Vascularization: {:.2f}%".format(y_proba_Class0)
  dme="Probability of the input image to have Diabetic Macular Edema: {:.2f}%".format(y_proba_Class1)
  drusen="Probability of the input image to have Hard Drusen: {:.2f}%".format(y_proba_Class2)
  normal="Probability of the input image to be absolutely normal: {:.2f}%".format(y_proba_Class3)

  total = [cnv,dme,drusen,normal]

  list_proba = [y_proba_Class0,y_proba_Class1,y_proba_Class2,y_proba_Class3]
  statements = ["Inference: The image has high evidence of Choroidal Neo Vascularization in the retinal pigment epithelium.",
               "Inference: The image has high evidence of Diabetic Macular Edema in the retinal pigment epithelium.",
               "Inference: The image has high evidence of Hard Drusen in the retinal pigment epithelium.",
               "Inference: The retinal pigment epithelium layer looks absolutely normal."]


  index = list_proba.index(max(list_proba))
  prediction = statements[index]

  return total, prediction
"""

"""
def translate_pnm(preds):
  y_proba_Class0 = preds.flatten().tolist()[0] * 100
  y_proba_Class1 = preds.flatten().tolist()[1] * 100
  y_proba_Class2 = preds.flatten().tolist()[2] * 100

  bac="Probability of the image to be Bacterial Pneumonia: {:.2f}%".format(y_proba_Class0)
  norm="Probability of the image to be Normal: {:.2f}%".format(y_proba_Class1)
  viral="Probability of the image to be Viral Pneumonia: {:.2f}%\n".format(y_proba_Class2)

  total = [bac,norm,viral]

  list_proba = [y_proba_Class0,y_proba_Class1,y_proba_Class2]
  statements = ["Inference: The chest X-Ray image shows high evidence of Bacterial Pneumonia.",
                "Inference: The chest X-Ray image is normal.",
                "Inference: The chest X-Ray image shows high evidence of Viral Pneumonia."]

  index = list_proba.index(max(list_proba))
  prediction = statements[index]

  return total, prediction
"""

"""
def translate_retinopathy(preds):
  y_proba_Class0 = preds.flatten().tolist()[0] * 100
  y_proba_Class1 = preds.flatten().tolist()[1] * 100
  y_proba_Class2 = preds.flatten().tolist()[2] * 100
  y_proba_Class3 = preds.flatten().tolist()[3] * 100

  mild="Probability of the input image to have Mild Diabetic Retinopathy: {:.2f}%".format(y_proba_Class0)
  mod="Probability of the input image to have Moderate Diabetic Retinopathy: {:.2f}%".format(y_proba_Class1)
  norm="Probability of the input image to be Normal: {:.2f}%".format(y_proba_Class2)
  severe="Probability of the input image to have Severe Diabetic Retinopathy: {:.2f}%".format(y_proba_Class3)

  total = [mild,mod,norm,severe]

  list_proba = [y_proba_Class0,y_proba_Class1,y_proba_Class2,y_proba_Class3]
  statements = ["Inference: The image has high evidence for Mild Nonproliferative Diabetic Retinopathy Disease.",
               "Inference: The image has high evidence for Moderate Nonproliferative Diabetic Retinopathy Disease.",
               "Inference: The image has no evidence for Nonproliferative Diabetic Retinopathy Disease.",
               "Inference: The image has high evidence for Severe Nonproliferative Diabetic Retinopathy Disease."]

  index = list_proba.index(max(list_proba))
  prediction = statements[index]

  return total, prediction

"""
