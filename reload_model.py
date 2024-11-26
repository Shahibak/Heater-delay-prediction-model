
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import TFSMLayer

############################## Inserting NN predictive model###############################

new_model = keras.models.load_model('dnn_model.keras')

##############################  Input parameters here:###############################

Bp = 17 # Magnet field
fcusc = 2.2 # Fraction of copper to superconductor
fraG = 0.15 # Fraction of G10
tG10 = 0.5 # Thickness of G10
tka = 0.1 # Thickness of kapton
tau = 12 # Time constant
Wcoil = 15 # Width of the coil
jhea = 350 # Heater power
Tcs = 8 # current sharing temperature

############################## Do not change the following section###############################

# Maximum and minimum value of each variable in the dataset
min_values = [0.8, 0.1, 0.025, 200.0, 0.1, 10.0, 12.73, 1.0, 5.0]
max_values = [2.2, 0.2, 0.075, 600.0, 0.22, 45.0, 20.0, 17.0, 16.0]

normalized_data = []

                        #fcusc   t_G10   t_kapton     jheater     G10_percentage     tau      W coil   B   TCS
non_normalized_data  = [ fcusc,     tG10,     tka,       jhea,          fraG,           tau ,      Wcoil,    Bp,  Tcs ]


for i in range(len(non_normalized_data)):
    # Apply min-max normalization formula
    j = (non_normalized_data[i] - min_values[i]) / (max_values[i] - min_values[i])
    # Append normalized value to the list
    normalized_data.append(j)

print (normalized_data)

####Prediction

# Convert the list to a NumPy array
non_normalized_array = np.array(normalized_data)


# Reshape the input array according to the model's input shape
# For example, if your model expects input shape (7,), reshape it to (1, 7)
# Adjust the shape according to your model's input requirements
input_data = non_normalized_array.reshape(1, -1)

# Make predictions using the model
predictions = new_model.predict(input_data)


print("Prediction value for heater delay is:", predictions)
