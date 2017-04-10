from clarifai.rest import ClarifaiApp
from pprint import pprint

app = ClarifaiApp("cRrYq4f5NYZw014zlOZOK3wzhZJF_a_MWGeVO3Ee", "tNkOF4tINoNzUgUVBHJR07E0Yb0t-LTwrhhT_QZX")

model = app.models.get('general-v1.3')

prediction = model.predict_by_url(url='https://samples.clarifai.com/metro-north.jpg')

pprint(prediction['outputs'][0])


