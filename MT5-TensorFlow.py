## MT5 TensorFlow

import tensorflow as tf
from transformers import TFT5ForConditionalGeneration, T5Tokenizer

# Specify the GPU device
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

# Initialize the model
model = TFT5ForConditionalGeneration.from_pretrained("google/mt5-large")

# Use a TensorFlow device context
with tf.device('/device:GPU:0'):
    tokenizer = T5Tokenizer.from_pretrained("google/mt5-large")

    # Create a large batch of inputs
    all_articles = ["UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."] * 80  # Adjust as needed

    # Specify batch size
    batch_size = 50

    # List to store all translations
    all_translations = []

    # Loop over articles in batches
    for i in range(0, len(all_articles), batch_size):
        articles = all_articles[i:i+batch_size]

        # Encode the source texts
        inputs = tokenizer(articles, padding=True, truncation=True, max_length=200, return_tensors="tf")

        # Translate the source texts
        translations = model.generate(inputs["input_ids"], num_beams=4, max_length=50, early_stopping=True)

        # Decode the translations
        translation_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in translations]

        # Add this batch's translations to the list of all translations
        all_translations.extend(translation_texts)

    print(all_translations[:5])  # Print the first 5 translations
