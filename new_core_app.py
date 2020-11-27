
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import keras.backend as K
import tensorflow
from datetime import datetime as dt
import numpy as np
import cv2
from cv2 import resize, INTER_AREA
import uuid
from PIL import Image
import os
import tempfile
from tensorflow.keras.models import load_model
import imageio
from tensorflow.keras.preprocessing import image

import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.imagenet_utils import decode_predictions

"""
def resize_image_tom(image):
    resized_image = cv2.resize(image, (256,256)) #Resize all the images to 128X128 dimensions
    if(len(resized_image.shape)!=3):
        resized_image = cv2.cvtColor(resized_image,cv2.COLOR_GRAY2RGB) #Convert to RGB
    return resized_image
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

@app.route('/tomato.html')
def tomato():
    return render_template('tomato.html')

@app.route('/grape.html')
def grape():
    return render_template('grape.html')

@app.route('/corn.html')
def crop():
    return render_template('corn.html')

@app.route('/potato.html')
def potato():
    return render_template('potato.html')

@app.route('/about.html')
def about():
    return render_template('about.html')


@app.route('/index.html')
def index_from_checkup():
    return render_template('index.html')

@app.route('/checkup.html')
def checkup_from_any():
    return render_template('checkup.html')



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

      if(type_=='tom' or type_=='grape' or type_=='corn' or type_=='potato'):
        test_image = image.load_img(name, target_size = (256, 256))
        test_image = image.img_to_array(test_image)
        test_image = test_image/255
        test_image = np.expand_dims(test_image, axis = 0)
        data=test_image




      #model=get_model(type_)[0]
      #model = load_model("static/weights/tomato.h5")
      #type_="tom"

      if(type_=='tom'):
         model=load_model("static/weights/tomato.h5")
         pred_val = translate_tomato(model.predict(data))
         final_json.append({"empty": False, "type":type_,
                            "pred_val": pred_val})
         #final_json.append({"empty": False, "type":type_,
                            #"pred_val": warn})

      elif(type_=='grape'):
           model=load_model("static/weights/grape.h5")
           pred_val = translate_grape(model.predict(data))
           final_json.append({"empty": False, "type":type_,
                              "pred_val": pred_val})

      elif(type_=='corn'):
           model=load_model("static/weights/corn.h5")
           pred_val = translate_corn(model.predict(data))
           final_json.append({"empty": False, "type":type_,
                              "pred_val": pred_val})

      elif(type_=='potato'):
           model=load_model("static/weights/potato.h5")
           pred_val = translate_potato(model.predict(data))
           final_json.append({"empty": False, "type":type_,
                              "pred_val": pred_val})



    else:
      warn = "Feeding blank image won't work. Please enter an input image to continue."
      pred_val =" "
      final_json.append({"pred_val": warn,"Tomato___Bacterial_spot":" ",
                            "Tomato___Early_blight":" ",
                            "Tomato___Late_blight":" ",
                            "Tomato___Leaf_Mold":" ",
                            "Tomato___Septoria_leaf_spot":" ",
                            "Tomato___Spider_mites Two-spotted_spider_mite":" ",
                            "Tomato___Target_Spot":" ",
                            "Tomato___Tomato_Yellow_Leaf_Curl_Virus":" ",
                            "Tomato___Tomato_mosaic_virus":" ",
                            "Tomato___healthy":" ",
                            })

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
models and the value will contain the request type."""
"""
def get_model(name = None):
  model_name = []
  if(name=='tom'):
    model_name.append({"model": load_model("static/weights/tomato.h5"), "type": name})
  elif(name=='grape'):
    model_name.append({"model": load_model_("grape.h5"), "type": name})
  #print(model_name)
  return model_name

"""


"""preds will contain the predictions made by the model. We will take the class probabalities and
store them in individual variables. We will return the class probabilities and the final predictions
made by the model to the frontend. The value contained in variables total and prediction will be
displayed in the frontend HTML layout."""
def translate_tomato(preds):
  dicti=["Bacterial Spot","Early Blight","Late Blight","Leaf Mold","Septoria Leaf Spot","Spider_mites Two-spotted_spider_mite","Target Spot","Tomato_Yellow_Leaf_Curl_Virus","Tomato mosaic virus","Healthy - your crop is fine , No Problem."]
  return dicti[np.argmax(preds)]


def translate_grape(preds):
  dicti=["Black_rot","Esca_(Black_Measles)","Leaf_blight_(Isariopsis_Leaf_Spot)","Healthy - your crop is fine , No Problem."]
  return dicti[np.argmax(preds)]

def translate_corn(preds):
  dicti=["Cercospora_leaf_spot Gray_leaf_spot","Common_rust_","Northern_Leaf_Blight","Healthy - your crop is fine , No Problem."]
  return dicti[np.argmax(preds)]


def translate_potato(preds):
  dicti=["Early_blight","Late_blight","Healthy - your crop is fine , No Problem."]
  return dicti[np.argmax(preds)]



  """list_proba = [y_proba_Class0, y_proba_Class1, y_proba_Class2, y_proba_Class3,y_proba_Class4,y_proba_Class5,y_proba_Class6,y_proba_Class7,y_proba_Class8,y_proba_Class9]
  statements = [
      "Inference: The image has high evidence of Bacterial Spot.",
      "Inference: The image has high evidence of Early Blight.",
      "Inference: The image has high evidence of Late Blight.",
      "Inference: The image has high evidence of leaf mold.",
      "Inference: The image has high evidence of Septoria Leaf Spot.",
      "Inference: The image has high evidence of Spider_mites Two-spotted_spider_mite.",
      "Inference: The image has high evidence of Target Spot.",
      "Inference: The image has high evidence of Tomato_Yellow_Leaf_Curl_Virus.",
      "Inference: The image has high evidence of Tomato mosaic virus.",
      "Inference: The image has high evidence of Healthy."]
  index = list_proba.index(max(list_proba))
  prediction = statements[index]

  return total, prediction"""

#Predict image using VGG16 pretrained models

"""
def convert_results(string):
  name = string.replace("_"," ")
  name = name.replace("-"," ")
  name = name.title()
  return name
"""




if __name__=="__main__":
  app.run("0.0.0.0",80, debug = False)
