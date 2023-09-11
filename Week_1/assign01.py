import IPython.display as ipd
import librosa
import librosa.display
import numpy as np

# filename = "/content/drive/MyDrive/ColabNotebooks/sound.wav" # google colab 용
filename = "sound.wav" 
data, sample_rate = librosa.load(filename)
ipd.Audio(filename)

cf = [0, 0, 0, 0, 0]


def func(cf):
    time = np.linspace(0, 0.5, len(data))
    sound = np.sum(cf[0] * np.sin(np.array([i * time for i in cf[1:]])), axis=0)
    return sound


def mse(func):
    error = np.abs(data - func())
    return np.mean(error) * 1500


def test():
    return func(cf)


# 여기서부터 코드를 작성하시오

init_cf = [max(data) / 4, 800, 1000, 1200, 1400]

# best_cf를 찾으시오

##################
# first attempt
best_cf = init_cf
err = 1000000000
eps = 100
trial = 140

for i in range(trial):
    print("trial " + str(i+1) + " : " + str(eps))

    c_t = np.array(best_cf.copy()) - eps
    c_t[0] += eps - (eps / 1000)

    for j in range(1, 5):
        for c in np.arange(0, eps * 2, eps / 20):
            cf = best_cf.copy()
            cf[j] = c_t[j] + c
            tmp = mse(test)
            if tmp < err:
                best_cf = cf
                err = tmp

    for c in np.arange(0, (eps/1000) * 2, eps / 100000):
        cf = best_cf.copy()
        cf[0] = c_t[0] + c
        tmp = mse(test)
        if tmp < err:
            best_cf = cf
            err = tmp
            # print(err, best_cf)

    eps /= 1.1
    print(err, best_cf)

cf = best_cf

err = mse(test)
print("final : " + str(err) + ", " + str(best_cf))
if err <= 5:
    print("pass the test")
else:
    print("fail the test")
##################

##################
# second attempt #


##################
