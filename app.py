from flask import Flask, render_template, request, url_for
from PIL import Image
from werkzeug.utils import secure_filename


import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np





app = Flask(__name__)
MODEL_PATH = 'E:\FlaskIntroduction\models\covid_classifier.pt'




resnet18 = torchvision.models.resnet18(pretrained=True)
resnet18.fc = torch.nn.Linear(in_features=512, out_features=3)
resnet18.load_state_dict(torch.load(MODEL_PATH,  map_location=torch.device('cpu')))
resnet18.eval()
    


test_transform = torchvision.transforms.Compose([
                 torchvision.transforms.Resize(size=(224, 224)),
                 torchvision.transforms.ToTensor(),
                 torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def predict_image_class(image_path):
    class_names = ['normal', 'viral', 'covid']

    image = image_path.convert('RGB')
    image = test_transform(image)
    image = image.unsqueeze(0)
    output = resnet18(image)[0]
    probabilities = torch.nn.Softmax(dim=0)(output)
    probabilities = probabilities.cpu().detach().numpy()
    predicted_class_index = np.argmax(probabilities)
    predicted_class_name = class_names[predicted_class_index]
    return probabilities, predicted_class_index, predicted_class_name



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_input',methods=['POST','GET'])
def get_input():
    if request.method == 'POST':
        img_file = request.files.get('image_name','not entered')
         if img_file.filename == '':
            return 'No File Detected'
        image = Image.open(img_file.stream)
        
        prob, index, label = predict_image_class(image)
       
        return render_template('result.html', label=label,prob=prob)
        
    else:
        return render_template('index.html')





if __name__ == '__main__':
    app.run(debug=True)
