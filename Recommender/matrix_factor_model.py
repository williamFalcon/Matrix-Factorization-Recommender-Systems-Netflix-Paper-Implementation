import numpy
import sys


def run_demo(args):
    print args
    data = args[1]
    model = ProductRecommender()
    model.fit(data)


class ProductRecommender(object):
    """
    Generates recommendations using the matrix factorization approach.
    Derived and implemented from the Netflix paper.

    Author: William Falcon

    Has 2 modes:
    Mode A: Derives P, Q matrices intrinsically for k features.
    Use this approach to learn the features.

    Mode B: Derives P matrix given a constant P matrix (Products x features). Use this if you want to
    try the approach of selecting the features yourself.

    Example 1:

    from matrix_factor_model import ProductRecommender
    modelA = ProductRecommender()
    data = [[1,2,3], [0,2,3]]
    modelA.fit(data)
    model.predict_instance(1)
    # prints array([ 0.9053102 ,  2.02257811,  2.97001565])

    Model B example
    modelB = ProductRecommender()
    data = [[1,2,3], [0,2,3]]

    # product x features
    Q = [[2,3], [2, 4], [5, 9]]

    # fit
    modelA.fit(data, Q)
    model.predict_instance(1)
    # prints array([ 0.9053102 ,  2.02257811,  2.97001565])

    """

    def __init__(self):
        self.Q = None
        self.P = None

    def fit(self, user_x_product, latent_features_guess=2, learning_rate=0.0002, steps=5000, regularization_penalty=0.02, convergeance_threshold=0.001):
        """
        Trains the predictor with the given parameters.
        :param user_x_product:
        :param latent_features_guess:
        :param learning_rate:
        :param steps:
        :param regularization_penalty:
        :param convergeance_threshold:
        :return:
        """
        print 'training model...'
        return self.__factor_matrix(user_x_product, latent_features_guess, learning_rate, steps, regularization_penalty, convergeance_threshold)

    def predict_instance(self, row_index):
        """
        Returns all predictions for a given row
        :param row_index:
        :return:
        """
        return numpy.dot(self.P[row_index, :], self.Q.T)

    def predict_all(self):
        """
        Returns the full prediction matrix
        :return:
        """
        return numpy.dot(self.P, self.Q.T)

    def get_models(self):
        """
        Returns a copy of the models
        :return:
        """
        return self.P, self.Q

    def __factor_matrix(self, R, K, alpha, steps, beta, error_limit):
        """
        R = user x product matrix
        K = latent features count (how many features we think the model should derive)
        alpha = learning rate
        beta = regularization penalty (minimize over/under fitting)
        step = logistic regression steps
        error_limit = algo finishes when error reaches this level

        Returns:
        P = User x features matrix. (How strongly a user is associated with a feature)
        Q = Product x feature matrix. (How strongly a product is associated with a feature)
        To predict, use dot product of P, Q
        """
        # Transform regular array to numpy array
        R = numpy.array(R)

        # Generate P - N x K
        # Use random values to start. Best performance
        N = len(R)
        M = len(R[0])
        P = numpy.random.rand(N, K)

        # Generate Q - M x K
        # Use random values to start. Best performance
        Q = numpy.random.rand(M, K)
        Q = Q.T

        error = 0

        # iterate through max # of steps
        for step in xrange(steps):

            # iterate each cell in r
            for i in xrange(len(R)):
                for j in xrange(len(R[i])):
                    if R[i][j] > 0:

                        # get the eij (error) side of the equation
                        eij = R[i][j] - numpy.dot(P[i, :], Q[:, j])

                        for k in xrange(K):
                            # (*update_rule) update pik_hat
                            P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])

                            # (*update_rule) update qkj_hat
                            Q[k][j] = Q[k][j] + alpha * ( 2 * eij * P[i][k] - beta * Q[k][j] )

            # Measure error
            error = self.__error(R, P, Q, K, beta)

            # Terminate when we converge
            if error < error_limit:
                break

        # track Q, P (learned params)
        # Q = Products x feature strength
        # P = Users x feature strength
        self.Q = Q.T
        self.P = P

        self.__print_fit_stats(error, N, M)

    def __error(self, R, P, Q, K, beta):
        """
        Calculates the error for the function
        :param R:
        :param P:
        :param Q:
        :param K:
        :param beta:
        :return:
        """
        e = 0
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:

                    # loss function error sum( (y-y_hat)^2 )
                    e = e + pow(R[i][j]-numpy.dot(P[i,:],Q[:,j]), 2)

                    # add regularization
                    for k in xrange(K):

                        # error + ||P||^2 + ||Q||^2
                        e = e + (beta/2) * ( pow(P[i][k], 2) + pow(Q[k][j], 2) )
        return e

    def __print_fit_stats(self, error, samples_count, products_count):
        print 'training complete...'
        print '------------------------------'
        print 'Stats:'
        print 'Error: %0.2f' % error
        print 'Samples: ' + str(samples_count)
        print 'Products: ' + str(products_count)
        print '------------------------------'

if __name__ == '__main__':
    run_demo(sys.argv)
