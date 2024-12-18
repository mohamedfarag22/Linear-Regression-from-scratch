import json
import numpy as np
import pickle
#import joblib
class LinearRegression():
    def __int__(self):
        self.weights = None
        self.bias = None

    def fit(self,x,y,lr=0.01,epoch=500):
        n_samples,n_features = x.shape
        self.weights = np.random.rand(n_features) #vector 1D (2,)
        self.bias = 0
        for iterate in range(epoch):
            y_prediction = np.dot(x,self.weights) + self.bias # x.shape = [3,2] * [2,] -> self.weights.shape => [3,1]
            dw = (1/n_samples) * np.dot(x.T,(y_prediction-y))
            db = (1/n_samples) * np.sum((y_prediction-y))
            self.weights-=lr * dw
            self.bias -= lr * db
    def predict(self,x):
        prediction = np.dot(x,self.weights) + self.bias
        return prediction
    def RootMeanSqear(self,y_truth,y_predict):
        return np.mean((y_truth - y_predict)**2)

    def save_model(self, file_name='LinearRegression.pkl'):
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)
        print(f'The model is saved in {file_name}')

    def save_parameters(self,file_name='LinearRegression.json'):
        parametes = {
            'weights' : self.weights.tolist(),
            'bias' : self.bias
        }
        with open(file_name,'w') as f :
            json.dump(parametes,f)
        print(f'The model parameter is saved in {file_name}')



model = LinearRegression()
x = np.array([[1,2],[2,3],[4,5]])
y = np.array([5,7,11])
model.fit(x, y)
prediction_model = model.predict(x)
the_loss = model.RootMeanSqear(y, prediction_model)
print(f'The prediction {prediction_model}')
print(f'The MSE of model is {the_loss}')
print(f'R^2 (Coefficient of determination): {1 - the_loss / np.var(y)}') #var(y) for variance of y to make balance in R2 and for variance data get optimal result
print(f'the weights {model.weights} and the bias is {model.bias}')
#-------------------------------------------------
#for save the model by pkl
# model.save_model('LinearRegression.pkl')
#-------------------------------------------------
#for load the all model form pkl file
# def load_model(file_name='LinearRegression.pkl'):
#     with open(file_name, 'rb') as f:
#         model = pickle.load(f)
#     print(f'The model is loaded from {file_name}')
#     return model
# #--------------------------------------------------
#for save the parameter model in json file
# model.save_parameters()
#--------------------------------------------------
#for upload the json file and print them
# file_name ='LinearRegression.json'
# with open(file_name,'r') as f :
#     parameters = json.load(f)
# print(parameters)

