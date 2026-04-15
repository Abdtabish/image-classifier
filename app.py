from flask import Flask,render_template,request
from PIL import Image
import torch
import torchvision.transforms as transforms
import pickle
import torch.nn as nn

app = Flask(__name__)

class CNNModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1=nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
    self.conv2=nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
    self.conv3=nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
    self.pooling=nn.MaxPool2d(kernel_size=2, stride=2)

    self.relu=nn.ReLU()
    self.flatten=nn.Flatten()
    self.linear=nn.Linear((128*16*16),128)
    self.output=nn.Linear(128,3)

  def forward(self, x):
      x = self.conv1(x) # -> Outputs: (32, 128, 128)
      x = self.pooling(x)# -> Outputs: (32, 64, 64)
      x = self.relu(x)
      x = self.conv2(x) # -> Outputs: (64, 64, 64)
      x = self.pooling(x) # -> Outputs: (64, 32, 32)
      x = self.relu(x)
      x = self.conv3(x) # -> Outputs: (128, 32, 32)
      x = self.pooling(x) # -> Outputs: (128, 16, 16)
      x = self.relu(x)
      x = self.flatten(x)
      x = self.linear(x)
      x = self.output(x)

      return x  


model = CNNModel()
model.load_state_dict(torch.load('animal_faces_model.pth',map_location='cpu'))
model.eval()
transform=transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float)
])
 
with open('label_enoder.pkl','rb') as f:
    le=pickle.load(f)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']   
    

    if file:
        img=Image.open(file).convert('RGB')
        img=transform(img)     
        
        img = img.unsqueeze(0)

        with torch.no_grad():
            output=model(img)
            output=torch.argmax(output,axis=1).item()

            result=le.inverse_transform([output])    


        return f"Prediction: {result}"

    return "No file uploaded"

if __name__ == "__main__":
    app.run(debug=True)
