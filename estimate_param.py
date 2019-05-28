from scipy import optimize
import numpy as np
import matplotlib.pyplot  as plt

#Sigmoid
def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


#Get probability
def get_prob(x, y, weight_vector):
    feature_vector =  np.array([x, y, 1])
    z = np.inner(feature_vector, weight_vector)
    return sigmoid(z)


#Define likelihood
def define_likelihood(weight_vector, *args):
    likelihood = 0
    df_data = args[0]

    for x, y, c in zip(df_data.x, df_data.y, df_data.c):
        prob = get_prob(x, y, weight_vector)

        i_likelihood = np.log(prob) if c==1 else np.log(1.0-prob)
        likelihood = likelihood - i_likelihood

    return likelihood


#Estimate_weight
def estimate_weight(df_data, initial_param):
    parameter = optimize.minimize(define_likelihood,
                                  initial_param,
                                  args=(df_data),
                                  method='Nelder-Mead')

    return parameter.x


#Draw Border
def draw_split_line(weight_vector):
    a,b,c = weight_vector
    x = np.array(range(-10,10,1))
    y = (a * x + c)/-b
    plt.plot(x,y, alpha=0.3)