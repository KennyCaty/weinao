
from routes import app
from flask import render_template, abort, redirect, flash, url_for, request, jsonify

from ict_models.model_load import main_model_load
from werkzeug.utils import secure_filename 
from ict_models.figure_mne_ica2 import mne_picture


'''  secure_filename
可以自动将 My movie.mov 转化成 My_movie.mov
将文件名带“/”的 如 etc/passwd 转化为etc_passwd
就是为了文件名不发生异常
'''
import os

@app.route('/predict', methods=['POST','GET'])
def predict():
    # data_path = 'data\wjl_02.csv'
    # predict_result = main_model_load(data_path)

    if request.method=='GET':
        return render_template('predict/predict.html', uploaded=False)
    elif request.method=="POST":
        f = request.files['file111']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        f.save(file_path)
        flash('上传成功', category='success')
        return render_template('predict/predict.html', uploaded=True, filename=f.filename)
    # return str(predict_result)


@app.route('/predict/run', methods={'POST'})
def predict_run():
    if request.method=='POST':
        filename=request.get_json()['filname']
        print(filename)
        file_path=os.path.join(app.config['UPLOAD_FOLDER'], filename)
        predict_result = main_model_load(file_path)
        print(predict_result)
        res = {
            '0': predict_result[0],
            '1': predict_result[1],
            '2': predict_result[2],
            '3': predict_result[3],
            '4': predict_result[4],
            '5': predict_result[5],
            '6': predict_result[6]
        }   

        
        # 绘图
        save_folder = mne_picture(file_path, 'static/mne_photos')
        res['save_forder'] = save_folder
        return jsonify(res)


# @app.route('/predict/details', methods={'POST'})
# def predict_details():
#     if request.method=="POST":
#         filename=request.get_json()['filname']
#         file_path = os.join(app.config['UPLOAD_FOLDER'], filename)
#         save_folder = mne_picture(file_path, 'static/mne_photos')
#         return render_template('predict/details.html', save_folder=save_folder)