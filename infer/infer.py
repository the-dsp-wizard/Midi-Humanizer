import numpy as np
import scipy
import matplotlib.pyplot as plt
import modelCoeffs

class MidiHumanizer:
    def __init__(self):
        stddev = 0.07213827389933028
        mean = 0

        lower_bound = -stddev
        upper_bound = stddev

        a = (lower_bound - mean) / stddev  # = -1
        b = (upper_bound - mean) / stddev  # = +1

        self.trunc_gauss = scipy.stats.truncnorm(a, b, loc=0, scale=stddev)
        self.x_ = np.zeros(5)
        self.y_ = np.zeros(6)

    def process(self, diff):
        l0 = np.tanh(np.dot(np.append(diff, self.x_), modelCoeffs.l0_x) + np.dot(self.y_, modelCoeffs.l0_y) + modelCoeffs.l0_b) * modelCoeffs.l0_oc
        l1 = np.tanh(np.dot(np.append(diff, self.x_), modelCoeffs.l1_x) + np.dot(self.y_, modelCoeffs.l1_y) + modelCoeffs.l1_b) * modelCoeffs.l1_oc
        l2 = np.tanh(np.dot(np.append(diff, self.x_), modelCoeffs.l2_x) + np.dot(self.y_, modelCoeffs.l2_y) + modelCoeffs.l2_b) * modelCoeffs.l2_oc
        l3 = np.tanh(np.dot(np.append(diff, self.x_), modelCoeffs.l3_x) + np.dot(self.y_, modelCoeffs.l3_y) + modelCoeffs.l3_b) * modelCoeffs.l3_oc
        out = l0 + l1 + l2 + l3 + modelCoeffs.bias
        out += self.trunc_gauss.rvs(1)[0]

        self.x_[4] = self.x_[3]
        self.x_[3] = self.x_[2]
        self.x_[1] = self.x_[0]
        self.x_[0] = diff

        self.y_[5] = self.y_[4]
        self.y_[4] = self.y_[3]
        self.y_[3] = self.y_[2]
        self.y_[2] = self.y_[1]
        self.y_[1] = self.y_[0]
        self.y_[0] = out

        return out

midi_human = MidiHumanizer()

y = np.zeros(1000)

for i in range(1000):
    y[i] = midi_human.process(0.25)

plt.plot(y)
plt.show()