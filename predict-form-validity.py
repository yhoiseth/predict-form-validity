from clarifai.rest import ClarifaiApp
from clarifai.rest import Image
from clarifai.rest import client
import os


app = ClarifaiApp(os.environ['CLARIFAI_APP_ID'], os.environ['CLARIFAI_APP_SECRET'])

try:
    trained_model = app.models.get('form')
except client.ApiError:
    path_to_training_images = 'images/training/'
    path_to_valid_images = path_to_training_images + 'valid/'
    path_to_invalid_images = '%sinvalid/' % path_to_training_images
    valid_image_filenames = os.listdir(path_to_valid_images)

    training_images = [
        Image(
            filename=path_to_invalid_images + 'invalid.png',
            concepts=['invalid'],
            not_concepts=['valid']
        )
    ]

    for valid_image_filename in valid_image_filenames:
        path_to_valid_image = path_to_valid_images + valid_image_filename
        valid_image = Image(
            filename=path_to_valid_image,
            concepts=['valid'],
            not_concepts=['invalid'],
        )

        training_images.append(valid_image)

    app.inputs.bulk_create_images(training_images)

    model = app.models.create(
        model_id='form',
        concepts=['valid', 'invalid'],
        concepts_mutually_exclusive=True,
    )
    model.train()

    trained_model = app.models.get('form')

prediction = trained_model.predict_by_filename('images/testing/valid9.png')

most_likely_concept = prediction['outputs'][0]['data']['concepts'][0]

print(
    'The form is %.2f percent likely to be %s.' % (
        most_likely_concept['value'] * 100,
        most_likely_concept['name'],
    )
)
