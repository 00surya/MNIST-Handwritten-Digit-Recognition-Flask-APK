from flask import Flask,render_template,redirect,jsonify
from flask.globals import request
# from services import api
import numpy as np
import joblib
import pickle
from keras.models import load_model
from numpy.core.records import array



class api:
    
    def LR(self,arr):

        lr_ = joblib.load('./services/lr_digit_rec.pkl')
        pred = lr_.predict([arr])
        return pred

    def SVM(self,arr):
        
        # with open('./services/svm_digit_model.pkl','rb') as f:
        #     svm_ = pickle.load(f)
        # arr[arr>0] = 1
        svm_ = joblib.load('./services/svm_digit_model.pkl')
        pred = svm_.predict([arr])
        return pred

    def CNN(self,arr):

        cnn_ = load_model('./services/CNN_digit_rec.h5')
        arr = arr.reshape(28,28)
        arr = arr.reshape(1,*(arr.shape),1)
        pred = np.argmax(cnn_.predict([arr]))
        return pred

    def MNB(self,arr):

        mnb_ = open('./services/mnb_digit_rec.pkl','rb')
        mnb_ = joblib.load(mnb_)
        pred = mnb_.predict([arr])
        return pred

    def KNN(self,arr):

        knn_ = joblib.load('./services/knn_digit_rec.pkl')
        pred = knn_.predict([arr])
        return pred
    
app = Flask(__name__)




@app.route('/')
def hello():
    return render_template("index.html")

@app.route('/digit',methods=['POST'])
def digit():
    if request.method == 'POST':
        arr = request.json['array']
        arr = np.array(arr)
        arr = arr*255.0 # Normalising value 0.-255.0
        cnn_pred = api().CNN(arr)
        lr_pred = api().LR(arr)
        mnb_pred = api().MNB(arr)
        svm_pred = api().SVM(arr)
        knn_pred = api().KNN(arr)
        print(lr_pred,cnn_pred,mnb_pred,svm_pred)
        # print(str(lr_pred))
#         return jsonify(mnb_res=mnb_pred,l_r_res=lr_pred,cnn_res=cnn_pred)
        return jsonify(l_r_res=str(lr_pred[0]),cnn_res=str(cnn_pred),mnb_res = str(mnb_pred[0]),svm_res=str(svm_pred[0]),knn_res=str(knn_pred[0]))
        # return(str(cnn_pred))
    return render_template("index.html")


if __name__ == '__main__':
    # debug = True
    app.run(debug=True)
