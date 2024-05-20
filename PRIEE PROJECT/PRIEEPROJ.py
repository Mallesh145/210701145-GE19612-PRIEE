import requests
from PIL import Image
from keras_preprocessing.image import load_img, img_to_array
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from IPython.display import display, Image as IPImage

# Load the model
model = load_model('FV.h5')

# Define labels
labels = {0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 5: 'capsicum', 6: 'carrot',
          7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger',
          14: 'grapes', 15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce',
          19: 'mango', 20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas', 25: 'pineapple',
          26: 'pomegranate', 27: 'potato', 28: 'raddish', 29: 'soy beans', 30: 'spinach', 31: 'sweetcorn',
          32: 'sweetpotato', 33: 'tomato', 34: 'turnip', 35: 'watermelon'}

fruits = ['Apple', 'Banana', 'Bell Pepper', 'Chilli Pepper', 'Grapes', 'Jalepeno', 'Kiwi', 'Lemon', 'Mango', 'Orange',
          'Paprika', 'Pear', 'Pineapple', 'Pomegranate', 'Watermelon']
vegetables = ['Beetroot', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Corn', 'Cucumber', 'Eggplant', 'Ginger',
              'Lettuce', 'Onion', 'Peas', 'Potato', 'Raddish', 'Soy Beans', 'Spinach', 'Sweetcorn', 'Sweetpotato',
              'Tomato', 'Turnip']

# Thresholds for nutrition values
nutrition_thresholds = {
    'Calories': 50,  # 50 kcal per 100 grams
    'Protein': 1.0,  # 1 gram per 100 grams
    'Fat': 0.5,      # 0.5 gram per 100 grams
    'Carbohydrate': 10.0  # 10 grams per 100 grams
}

# List of more nutritious alternatives
nutritious_alternatives = {
    'apple': 'banana',
    'banana': 'mango',
    'carrot': 'sweetpotato',
    'lettuce': 'spinach',
    'cucumber': 'tomato',
    'bell pepper': 'paprika',
    'grapes': 'pomegranate'
}

# Your USDA API key here
API_KEY = 'FgafNBR8VPucQBKyp3xbJqLLbrXLAaMd6ZvKV9yL'

def fetch_nutrition(prediction):
    try:
        url = f'https://api.nal.usda.gov/fdc/v1/foods/search?query={prediction}&api_key={API_KEY}'
        response = requests.get(url)
        data = response.json()

        if 'foods' in data and len(data['foods']) > 0:
            food = data['foods'][0]
            calories = food.get('foodNutrients', [])[3]['value']  # Adjust index based on API response
            nutrition_info = {
                'Calories': calories,
                'Protein': food.get('foodNutrients', [])[0]['value'],  # Adjust index based on API response
                'Fat': food.get('foodNutrients', [])[1]['value'],      # Adjust index based on API response
                'Carbohydrate': food.get('foodNutrients', [])[2]['value']  # Adjust index based on API response
            }
            return calories, nutrition_info
        else:
            return None, {}
    except Exception as e:
        print("Can't fetch the nutrition information")
        print(e)
        return None, {}

def processed_img(img_path):
    img = load_img(img_path, target_size=(224, 224, 3))
    img = img_to_array(img)
    img = img / 255
    img = np.expand_dims(img, axis=0)
    answer = model.predict(img)
    y_class = answer.argmax(axis=-1)[0]
    res = labels[y_class]
    return res.capitalize()

def check_nutrition(nutrition_info):
    for key, threshold in nutrition_thresholds.items():
        if key in nutrition_info:
            if nutrition_info[key] < threshold:
                return False
    return True

def run(img_file):
    img = Image.open(img_file).resize((150, 150))
    display(IPImage(img_file))

    result = processed_img(img_file)
    print(f"Predicted: {result}")

    if result in vegetables:
        print('Category: Vegetables')
    else:
        print('Category: Fruit')

    calories, nutrition_info = fetch_nutrition(result)
    if calories:
        print(f'Calories: {calories} kcal (100 grams)')
    if nutrition_info:
        for key, value in nutrition_info.items():
            print(f'{key}: {value} g')

    if not check_nutrition(nutrition_info):
        if result.lower() in nutritious_alternatives:
            alternative = nutritious_alternatives[result.lower()]
            print(f'The {result} is low in nutritional value. Consider eating {alternative.capitalize()} for better nutrition.')

# Specify the path to your image here
img_file = 'Image_2.jpg'
run(img_file)
