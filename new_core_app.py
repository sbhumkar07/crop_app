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
      final_json.append({"pred_val": warn,
                            })

    K.clear_session()
    return jsonify(final_json)
  return jsonify({"empty":True})


def translate_tomato(preds):
  dicti=["Bacterial Spot - जीवाणूजण्य करपा SOLUTION - Prune plants to promote air circulation. Spraying with a copper fungicide will give fairly good control the bacterial disease." ,"Early Blight - लवकर येणारा करपा SOLUTION If disease is severe enough to warrant chemical control, select one of the following fungicides: mancozeb (very good); chlorothalonil or copper fungicides (good) ","Late Blight - उशीरा येणारा करपा SOLUTION If disease is severe enough to warrant chemical control, select one of the following fungicides: chlorothalonil (very good); copper fungicide, or mancozeb (good).","Leaf Mold - पानावरील कुरशी SOLUTION Using a preventative fungicide program with chlorothalonil, mancozeb or copper fungicide, can control the disease","Septoria Leaf Spot - सेप्टोरीया पानावरील ठिपके SOLUTION Repeated fungicide applications with chlorothalonil (very good) or copper fungicide, or mancozeb (good) will keep the disease in check ","Spider_mites Two-spotted_spider_mite - ठिपकेदार कोळी","Target Spot - पानावरील ठिपके","Tomato_Yellow_Leaf_Curl_Virus - टोमॅटोवरील पिवळा विषाणू SOLUTION Low concentration sprays of a horticultural oil or canola oil will act as a whitefly repellent, reduce feeding and possibly transmission of the virus. Use a 0.25 to 0.5% oil spray (2 to 4 teaspoons horticultural or canola oil & a few drops of dish soap per gallon of water) weekly","Tomato mosaic virus - टोमॅटो मोसँक विषाणू SOLUTION There are no chemical controls for viruses. Remove and destroy infected plants promptly. Wash hands thoroughly after smoking (the Tobacco mosaic virus may be present in certain types of tobacco) and before working in the garden. Eliminate weeds in and near the garden.","Healthy - your crop is fine , No Problem - तंदुरुस्त आहे"]
  return dicti[np.argmax(preds)]


def translate_grape(preds):
  dicti=["Black_rot - काळे ठिपके SOLUTION Mancozeb, and Ziram are all highly effective against black rot. Because these fungicides are strictly protectants, they must be applied before the fungus infects or enters the plant. They protect fruit and foliage by preventing spore germination.","Esca_(Black_Measles) SOLUTION Management starts with preventive practices in nursery mother blocks and propagation beds. Good management techniques, which include proper planting, irrigation, and fertility for young vines while avoiding devigorating stresses, both before and after planting, are very important for establishing a healthy and productive vineyard","Leaf_blight_(Isariopsis_Leaf_Spot) - पानावरील करपा SOLUTION Spraying of the grapevines at 3-4 leaf stage with fungicides like Bordeaux mixture @ 0.8% or Copper Oxychloride @ 0.25% or Carbendazim @ 0.1% are effective against this disease","Healthy - your crop is fine , No Problem - तंदुरुस्त आहे"]
  return dicti[np.argmax(preds)]

def translate_corn(preds):
  dicti=["Cercospora_leaf_spot Gray_leaf_spot  - सर्कोस्पोरा पानावरील ठिपके, तपकिरी पानावरील ठिपके  SOLUTION Disease management tactics include using resistant corn hybrids, conventional tillage where appropriate, and crop rotation. Foliar fungicides can be effective if economically warranted.","Common_rust_ - तांबरा SOLUTION The use of resistant hybrids is the primary management strategy for the control of common rust. Timely planting of corn early during the growing season may help to avoid high inoculum levels or environmental conditions that would promote disease development. Symptoms of common rust often appear after silking","Northern_Leaf_Blight - नोरदन पानावरील करपा SOLUTION Northern corn leaf blight can be managed through the use of resistant hybrids. Additionally, timely planting can be useful for avoiding conditions that favor the disease. The tan lesions of northern corn leaf blight are slender and oblong tapering at the ends ranging in size between 1 to 6 inches","Healthy - your crop is fine , No Problem - तंदुरुस्त आहे"]
  return dicti[np.argmax(preds)]


def translate_potato(preds):
  dicti=["Early_blight - लवकर येणारा करपा SOLUTION Treatment of early blight includes prevention by planting potato varieties that are resistant to the disease; late maturing are more resistant than early maturing varieties. Avoid overhead irrigation and allow for sufficient aeration between plants to allow the foliage to dry as quickly as possible","Late_blight - उशीरा येणारा करपा SOLUTION The severe late blight can be effectively managed with prophylactic spray of mancozeb at 0.25% followed by cymoxanil+mancozeb or dimethomorph+mancozeb at 0.3% at the onset of disease and one more spray of mancozeb at 0.25% seven days after application of systemic fungicides in West Bengal.","Healthy - your crop is fine , No Problem - तंदुरुस्त आहे"]
  return dicti[np.argmax(preds)]





if __name__=="__main__":
  app.run("0.0.0.0",5000, debug = False)
