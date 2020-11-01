import pickle, gzip, numpy as np
import matplotlib.pyplot as plt


def citire_date():
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_set, valid_set, test_set = pickle.load(f, encoding='latin')
    f.close()
    return training_set, valid_set, test_set


def activation(input):
    vect=[]
    for i in input.flatten():
        if i > 0:
            vect.append(1)
        else:
            vect.append(0)

    return vect


def percept_training_alg(w,b, eta, nr_iterations, training_set):
    xs,ts=training_set

    all_classified = False
    while not all_classified and nr_iterations > 0:
        all_classified=True
        print(nr_iterations)
        for i in range(len(xs)):
            x=xs[i]
            tt=ts[i]
            t=[0,0,0,0,0,0,0,0,0,0]
            t[tt]=1

            z = w.dot(x) + b
            output = activation(z)
            x=np.asarray([x])
            t=np.asarray([t])
            output=np.asarray([output])

            w = w + (t - output).T.dot(x) * eta
            b = b + (t - output) * eta
            if output is not t:
                all_classified = False
        nr_iterations -= 1
    return w,b


def train(training_set):
    w = np.random.random((10,784))
    b= np.random.random(10)

    w,b=percept_training_alg(w,b,0.0002,100,training_set)
    np.save("nn_w",w)
    np.save("nn_b",b)
    return w,b

def validate(w,b,valid_set):
    corecte=0
    gresite=0
    xs,ts=valid_set
    # print(ts[:10])
    for i in range(len(xs)):
        x = xs[i]
        tt = ts[i]
        z = w.dot(x) + b
        pos,m = max(enumerate(z.flatten()),key=lambda x:x[1])
        '''
        if i<10:
            print(z)
            print(pos)
        '''
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
w,b=train(training_set)
display(w[0])
print(validate(w,b,valid_set))