from PIL import Image
import numpy as np
import joblib

img = Image.open('number3.png')

data = list(img.getdata(band=2))
for i in range(len(data)):
    data[i] = 255 - data[i]
three = np.array(data)/256
model = joblib.load('hd_recognition.sav')
p = model.predict([three])
print(p)
