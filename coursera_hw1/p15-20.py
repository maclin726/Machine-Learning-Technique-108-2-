from libsvm.svmutil import *
import random
import math

with open('features.train', 'r') as f:
    lines = f.readlines()
    train_data = [ [ float(element) for element in line.split(' ') if element != ''] for line in lines ]

with open('features.test', 'r') as f:
    lines = f.readlines()
    test_data = [ [ float(element) for element in line.split(' ') if element != ''] for line in lines ]

train_x, test_x = [ [d[1], d[2]] for d in train_data ], [ [d[1], d[2]] for d in test_data ]
train_label, test_label = [ d[0] for d in train_data ], [ [d[1], d[2]] for d in test_data ]


def p15():
    train_y = [ -1 if l != 0 else +1 for l in train_label ]
    prob = svm_problem(train_y, train_x)
    param = svm_parameter('-t 1 -d 1 -c 0.01 -h 0')
    m = svm_train(prob, param)

    w_1, w_2 = 0, 0
    for sv, coef in zip(m.get_SV(), m.get_sv_coef()):
        w_1 += coef[0] * sv[1]
        w_2 += coef[0] * sv[2]

    print(w_1, w_2)
    print((w_1**2 + w_2**2)**0.5)

def p16_17():
    digits = [0, 2, 4, 6, 8]
    for digit in digits:
        print(f'------\nPredicting {digit}')
        train_y = [ -1 if l != digit else +1 for l in train_label ]
        prob = svm_problem(train_y, train_x)
        param = svm_parameter('-t 1 -d 2 -c 0.01 -h 0')
        m = svm_train(prob, param)
        p_labels, p_acc, p_vals = svm_predict(train_y, train_x, m)
        print( 'sum of alphas is ', sum([abs(alpha[0]) for alpha in m.get_sv_coef()]) )


def p19():
    gamma = [1, 10, 100, 1000, 10000]
    train_y = [ -1 if l != 0 else +1 for l in train_label ]
    test_y = [ -1 if l != 0 else +1 for l in test_label ]
    prob = svm_problem(train_y, train_x)
    
    for g in gamma:
        print(f'------\nPredicting gamma = {g}')
        param = svm_parameter(f'-t 2 -g {g} -c 0.1 -h 0')
        m = svm_train(prob, param)
        p_labels, p_acc, p_vals = svm_predict(test_y, test_x, m)


def p20():
    gamma = [1, 10, 100, 1000, 10000]
    val_acc = {1: 0, 10: 0, 100: 0, 1000: 0, 10000: 0}
    
    train_size = len(train_x)
    for time in range(1000):
        print(time)
        val_indices = random.sample(list(range(train_size)), 1000)
        val_x = [ train_x[i] for i in range(train_size) if i in val_indices ]
        val_y = [ -1 if train_label[i] != 0 else +1 for i in range(train_size) if i in val_indices ]
        new_train_x = [ train_x[i] for i in range(train_size) if i not in val_indices ]
        new_train_y = [ -1 if train_label[i] != 0 else +1 for i in range(train_size) if i not in val_indices ]
        prob = svm_problem(new_train_y, new_train_x)

        for g in gamma:
            param = svm_parameter(f'-t 2 -g {g} -c 0.1 -h 0')
            m = svm_train(prob, param)
            p_labels, p_acc, p_vals = svm_predict(val_y, val_x, m)
            val_acc[g] += p_acc[0]

    print(val_acc)
    # {1: 8938.3, 10: 9009.799999999997, 100: 8988.099999999997, 1000: 8367.199999999999, 10000: 8367.199999999999}

p16_17()