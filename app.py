from flask import Flask,render_template,request
import os
from tooth_detection import run_model

app = Flask(__name__)
@app.route("/upload",methods=['GET', 'POST'])
def index():

    if request.method=="POST":
        # file = request.files['image']
        # upload_folder = '/Users/krishna/Desktop/aiDent/images'
        # os.makedirs(upload_folder, exist_ok=True)
        # file_path = os.path.join(upload_folder, file.filename)
        # file.save(file_path)
        file = request.files['image']
        upload_folder = 'static/images'
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)
        print(run_model(file_path))

        # Optionally, you can return a response to the user
        return "File uploaded successfully"

    return render_template("input.html")


if __name__ == '__main__':
    app.run(debug=True,port=5300)


