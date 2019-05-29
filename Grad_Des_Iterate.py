import numpy as np
import matplotlib.pyplot as plt

def get_sq_diff(f1_bef,f1_aft,l):
    sub_sq = [0,0]
    sub_sq[0] = (sum(f1_bef[l*0:l*1]) - sum(f1_aft[l*0:l*1]))**2
    sub_sq[1] = (sum(f1_bef[l*1:l*2]) - sum(f1_aft[l*1:l*2]))**2
    w1_ans = sub_sq[1]/(sub_sq[0]+sub_sq[1])
    print("Answer: ", w1_ans)
    return sub_sq[:]

def f1(x, sub_sq):
    return 1/2*sub_sq[0]*x**2 + 1/3*sub_sq[1]*(1-x)**2

def f1_prime(x, sub_sq):
    return (sub_sq[0]+sub_sq[1])*x - sub_sq[1]

# 最急降下法
def gradient_descent_method(f_prime, init_x, learning_rate, sub_sq):
    eps = 1e-10

    x = init_x
    x_history = [init_x]
    iteration_max = 1000

    # 収束するか最大試行回数に達するまで
    for i in range(iteration_max):

        print(i+1, ":", x)

        # 最急上昇法の場合は-を+にする
        x_new = x + learning_rate * f_prime(x, sub_sq)
        if x_new > 0.9:
            x_new = 0.9
        if x_new < 0.1:
            x_new = 0.1

        # 収束条件を満たせば終了
        if abs(x - x_new) < eps:
            break

        x = x_new
        x_history.append(x)

    return (x, np.array(x_history))

# Draw raw signal
def draw_raw(fig,f1_bef,f1_aft,d):
    ax0 = fig.add_subplot(111)
    c0 = ax0.plot(d, f1_bef[:], label="Before")
    c1 = ax0.plot(d, f1_aft[:], label="After")
    ax0.set_title('Raw signal', fontsize=16)
    ax0.set_xlabel("Depth", fontsize=16)
    ax0.set_ylabel("Intensity", fontsize=16)
    ax0.legend()
    c2 = ax0.plot([4,4],[0,9], "black", linestyle="dashed")

def main():

    d = np.linspace(1, 7, 60)
    #Define two functions
    f1_bef = -10.429 + 26.44*d - 12.917*d**2 + 2.2929*d**3 - 0.1364*d**4
    f1_aft = -0.1429 + 2.3225*d + 0.9167*d**2 - 0.4394*d**3 + 0.0379*d**4

    l = int(np.size(d,0)/2)  # Divide by the number of segment
    sub_sq = get_sq_diff(f1_bef, f1_aft, l)

    fig0 = plt.figure(0)
    draw_raw(fig0, f1_bef, f1_aft, d)

    learning_rates = [ 0.000001, 0.00001, 0.0001]
    fig1 = plt.figure(figsize=(6, 8))

    for i, learning_rate in enumerate(learning_rates):
        ans, x_history = gradient_descent_method(f1_prime, 0.5, learning_rate, sub_sq)

        # subplotの場所を指定
        ax1 = plt.subplot(3, 1, (i+1)) # 3行2列の意味

        # グラフのタイトル
        ax1.set_title("learning rate: " + str(learning_rate) + ", iteration: " + str(len(x_history)))

        # グラフを描く
        x = np.arange(-0.3, 1.0, 0.01)
        y = f1(x, sub_sq)
        ax1.plot(x, y)

        # 移動した点を表示
        ax1.plot(x_history, f1(x_history, sub_sq), 'o')

        # 点同士を線で結ぶ
        for i in range(len(x_history)-1):
            x1 = x_history[i]
            y1 = f1(x_history[i], sub_sq)
            x2 = x_history[i+1]
            y2 = f1(x_history[i+1], sub_sq)
            ax1.plot([x1, x2], [y1, y2], linestyle='-', linewidth=2)

    # タイトルが重ならないようにする
#    plt.tight_layout()

    # 画像保存用にfigを取り出す
    fig = plt.gcf()
    # fig.set_size_inches(50.0, 60.0)

    # 画像を表示
    plt.show()

    # 画像を保存
    fig.savefig('gradient_descent_method.png')

main()