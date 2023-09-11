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
err = 1000000000    # 초기 에러값
eps = 100   # 탐색 범위
eps2 = 1.1
trial = 160 # 시도횟수

for i in range(trial):
    c_t = np.array(best_cf.copy()) - eps
    c_t[0] += eps - (eps / 1000)

    for j in range(1, 5):
        # eps를 줄여가며 best_cf 각 요소 +- eps 범위를 eps / 20 만큼 증가시키며 탐색
        for c in np.arange(0, eps * 2, eps / 20):
            cf = best_cf.copy()
            cf[j] = c_t[j] + c
            tmp = mse(test)
            if tmp < err:
                best_cf = cf
                err = tmp

    # c0는 따로 실행 -> c1, c2, c3, c4와 범위가 다르기 때문에
    for c in np.arange(0, (eps/1000) * 2, eps / 100000):
        cf = best_cf.copy()
        cf[0] = c_t[0] + c
        tmp = mse(test)
        if tmp < err:
            best_cf = cf
            err = tmp
            # print(err, best_cf)

    print("trial " + str(i+1) + " : " + str(eps))
    print(err, best_cf)
    eps /= eps2

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
