import numpy as np

import tensorflow as tf
from tensorflow.python.framework import ops

sess = tf.Session()


def gen_dima_list(x, session=None):
    # inputs is a tensor.
    # make sure `x` is a vector
    assert x.shape.ndims == 1
    x_eval = x.eval(session=session)

    total_points = x.shape[0]
    flag = None
    if x_eval[-2] > x_eval[-1]:
        flag = True
    elif x_eval[-2] < x_eval[-1]:
        flag = False
    else:
        # TODO: deal with x[-2] == x[-1]
        pass

    k = 1
    if flag is True:
        while x_eval[-1 - k] >= x_eval[-1]:
            if k + 1 > total_points:
                print("iterate all the x points")
                break

            k += 1
    else:
        while x_eval[-1 - k] <= x_eval[-1]:
            if k + 1 > total_points:
                print("iterate all the x points")
                break
            k += 1

    # tailor the x
    x_eval = x_eval[-1-k:]

    # use *dima* strategy to generate data points
    brown_points = [x_eval[0]]
    index = 0
    if flag is True:
        find_maximum = True
        while index + 2 < x_eval.shape[0]:
            if find_maximum is True:
                next_index = np.argmax(x_eval[index:-1])
            else:
                next_index = np.argmin(x_eval[index:-1])
            brown_points.append(x_eval[index+next_index])
            index += next_index
            find_maximum = not find_maximum

    # import ipdb; ipdb.set_trace()
    if flag is False:
        find_minimum = True
        while index + 2 < x_eval.shape[0]:
            if find_minimum is True:
                next_index = np.argmin(x_eval[index:-1])
            else:
                next_index = np.argmax(x_eval[index:-1])
            brown_points.append(x_eval[index+next_index])
            index += next_index
            find_minimum = not find_minimum

    brown_points.append(x_eval[-1])

    brown_points = ops.convert_to_tensor(brown_points)

    return brown_points


if __name__ == "__main__":
    a = [-10, 9, -8, 7, -6, 5, -4, 3, -2, 1, -7]
    x = tf.constant(a, dtype=tf.float32)
    print(sess.run(gen_dima_list(x, sess)))

    a = [-10, -5. -3, -7, -2, -8, 0,  9, 7, -4, 7, -6, 5, -4, 3, -2, -8.5, 1, -9]
    x = tf.constant(a, dtype=tf.float32)
    print(sess.run(gen_dima_list(x, sess)))


    a = [-10, 9.5, -8, 7, -6, 5, -4, 3, -2, 9]   #
    x = tf.constant(a, dtype=tf.float32)
    print(sess.run(gen_dima_list(x, sess)))

    # a = [-10, -5. -3, -7, -2, -8, 0,  9, 7, -4, 7, -6, 5, -4, 3, -2, -8.5, 1, -9]
    # x = tf.constant(a, dtype=tf.float32)
    # print(sess.run(gen_dima_list(x, sess)))
