
import matplotlib.pyplot as plt #Para visualizacao dos dados e do erro
import numpy as np #Biblioteca de manipulacao de arrays Numpy
from matplotlib.colors import ListedColormap #Lista de cores para plotagens
import pandas as pd


### Carregar iris dataset
df = pd.read_csv('Dados_Treinamento_Sinal.csv',header=None)
df.head()

x = df.iloc[0:35,[0,1,2,3]].values
y = df.iloc[0:35,4].values

### Assumindo que Setosa(0) seja -1 e Versicolor = 1

### Plotar o grafico
### vermelhos ----> Classe2 (-1)
### azuis ----> Classe1 (1)

def plot(x,y):
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    plt.figure(figsize=(7,5))
    plt.scatter(x[:,0], x[:,1], c=y, cmap=cm_bright)
    plt.scatter(None, None, color = 'r', label='Setosa')
    plt.scatter(None, None, color = 'b', label='Versicolor')
    plt.legend()
    plt.title('Visualização do Dataset')
    plt.show()

plot(x,y)

###Construindo Adaline
class Adaline(object):
    def __init__(self, eta = 0.001, epoch = 100):
        self.eta = eta
        self.epoch = epoch

    def fit(self, x, y):
        np.random.seed(16)
        self.weight_ = np.random.uniform(-1, 1, x.shape[1] + 1)
        self.error_ = []
        
        cost = 0
        for _ in range(self.epoch):
            
            output = self.activation_function(x)
            error = y - output
            
            self.weight_[0] += self.eta * sum(error)
            self.weight_[1:] += self.eta * x.T.dot(error)
            
            cost = 1./2 * sum((error**2))
            self.error_.append(cost)
        
        return self

    def net_input(self, x):
        """Calculo da entrada z"""
        return np.dot(x, self.weight_[1:]) + self.weight_[0]
    def activation_function(self, x):
        """Calculo da saida da funcao g(z)"""
        return self.net_input(x)
    def predict(self, x):
        """Retornar valores binaros 0 ou 1"""
        return np.where(self.activation_function(x) >= 0.0, 1, -1)


###Plotando erros apos 100 epocas
names = ['Taxa de Aprendizado = 0.001', 'Taxa de Aprendizado = 0.01']
classifiers = [Adaline(), Adaline(eta = 0.01)]
step = 1
plt.figure(figsize=(14,5))
for name, classifier in zip(names, classifiers):
    ax = plt.subplot(1, 2, step)
    clf = classifier.fit(x, y)
    ax.plot(range(len(clf.error_)), clf.error_)
    ax.set_ylabel('Error')
    ax.set_xlabel('Epoch')
    ax.set_title(name)

    step += 1

plt.show()


### Plotando as fronteiras de decisao com Adaline
clf = Adaline()
clf.fit(x, y)

A = [0.4329,-1.3719,0.7022,-0.8535] # Classe 1
B = [0.3024,0.2286,0.8630,2.7909] #Classe -1

print (clf.predict(A))

print (clf.predict(B))
