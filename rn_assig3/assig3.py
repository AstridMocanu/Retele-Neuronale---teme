import pickle, gzip, numpy as np
import matplotlib.pyplot as plt
import time

#CITIRE
def citire_date():
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_set, valid_set, test_set = pickle.load(f, encoding='latin')
    f.close()
    return training_set, valid_set, test_set

#ACTIVATION FUNCTIONS
def sigmoid(input):
      return 1 / (1 + np.exp(-input))


def softmax(input):

    e_x = np.exp(input - np.max(input))
    return e_x / e_x.sum()


def init1():
    #Weights init to avoid saturation
    #ideea: np.random.randn(size_l, size_l-1)
    #standard normal distribution with mean 0 and standard deviation 1
    w = np.random.random((100,784))-0.5
    b= np.random.random((100,1))-0.5

    return w,b


def init2():
    w = np.random.random((10, 100))-0.5
    b = np.random.random((10,1))-0.5

    return w, b

def backpropagation_alg(w1,b1, w2,b2,eta,lbd,miu, nr_iterations, training_set,activations):
    xs, ts = training_set

    weights=[w1,w2]
    biases=[b1,b2]
    vw=[np.zeros(w1.shape),np.zeros(w2.shape)]
    vb=[np.zeros(b1.shape),np.zeros(b2.shape)]
    while nr_iterations > 0:


        start=time.time()
        print(nr_iterations)
        for i in range(len(xs)):

            deltas = []
            ys = []

            if i%10000==0: print(i)
            x = xs[i]
            tt = ts[i]
            t = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            t[tt] = 1
            ys.append(x)


            x=x.reshape(784,1)
            z = w1.dot(x) + b1
            output = activations[0](z)
            ys.append(output)


            x=output
            z = w2.dot(x)+b2
            output = activations[1](z)
            ys.append(output)



            t = np.asarray([t])


            #compute the error for the final layer
           # deltas.append(output*(1-output)*(output-t.T))

            #crossentropy
            deltas.append(output-t.T)



            nr_layers=2
            it=-1
            while nr_layers>0:
                #print(nr_layers,it)


                y_grec=ys[it - 1]
                # actualize the error for prev layer
                sum = weights[it].T.dot(deltas[-1])



                # modify the w and b for prev layer

                vw[it]=miu*vw[it]-eta * deltas[-1].dot(y_grec.reshape(1,-1))
                vb[it]=miu*vb[it]-eta * deltas[-1]
                #weights[it] -= eta * (deltas[-1].dot(y_grec.reshape(1,-1))+lbd*weights[it])
                weights[it] += vw[it]-eta * lbd*weights[it]
                #biases[it] -= eta * (deltas[-1]+lbd*biases[it])
                biases[it] += vb[it]-eta *lbd*biases[it]

                if nr_layers>1:
                    deltas.append(y_grec * (1 - y_grec) * sum)


                nr_layers-=1
                it-=1


        nr_iterations -= 1
        final=time.time()
        print(final-start)

    return w1,b1,w2,b2


#TRAIN NETWORK
def network(training_set):
    #3 layers:784 input, 100 hidden - sigmoid, 10 output - softmax
    w1,b1=init1()
    w2,b2=init2()
    activations=[]
    activations.append(sigmoid)
    activations.append(softmax)
    w1,b1,w2,b2=backpropagation_alg(w1,b1,w2,b2,0.002,0.001,0.9,10,training_set,activations)

    return w1,b1,w2,b2

#VALIDATE
def validate(w1,b1,w2,b2,valid_set):
    corecte=0
    gresite=0
    xs,ts=valid_set

    for i in range(len(xs)):
        x = xs[i]
        tt = ts[i]
        x = x.reshape(784, 1)
        z1 = w1.dot(x) + b1

        z2=sigmoid(z1)

        z2 = w2.dot(z2) + b2


        pos,m = max(enumerate(z2.flatten()),key=lambda x:x[1])

        if pos==tt:
            corecte+=1
        else:
            gresite+=1
    return corecte,corecte/(corecte+gresite)


def display(w):
    img=w.reshape(28,28)
    plt.matshow(img)
    plt.show()






training_set, valid_set, test_set = citire_date()
w1,b1,w2,b2=network(training_set)
#display(w1[0])
print(validate(w1,b1,w2,b2,valid_set))