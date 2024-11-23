import pickle
import numpy as np

filename2 = 'heart-model.pkl'
classifier2 = pickle.load(open(filename2, 'rb'))

data = np.array([[18393,2,168,62.0,110,80,1,1,0,0,1]])
my_prediction = classifier2.predict(data)
print(my_prediction[0])


