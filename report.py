import tensorflow as tf
from tensorflow.python.data import AUTOTUNE
import tensorflow_text

dataset_name = 'stonks'
saved_model_path = './{}_bert'.format(dataset_name.replace('/', '_'))
reloaded_model = tf.saved_model.load(saved_model_path)

test_ds = tf.keras.utils.text_dataset_from_directory(
    'data\\test',
    labels='inferred',
    batch_size=32)

test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)


def print_my_examples(inputs, results):
    result_for_printing = \
        [f'input: {inputs[i]:<30} : score: {results[i][0]:.6f}'
         for i in range(len(inputs))]
    print(*result_for_printing, sep='\n')
    print()


examples = [
    'this is such an amazing movie!',  # this is the same sentence tried earlier
    'The movie was great!',
    'The movie was meh.',
    'The movie was okish.',
    'The movie was terrible...'
]
reloaded_results = reloaded_model(tf.constant(examples))
print('Results from the saved model:')
print_my_examples(examples, reloaded_results)
