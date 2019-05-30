from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

# Square Difference for Step A (two segments)
def get_sq_diff_A(f1_bef,f1_aft, f1_10m,l):
    sub_sq = [0,0,0,0]
    sub_sq[0] = (sum(f1_bef[l*0:l*1]) - sum(f1_aft[l*0:l*1]))**2  # Difference between before and after
    sub_sq[1] = (sum(f1_bef[l*1:l*2]) - sum(f1_aft[l*1:l*2]))**2
    sub_sq[2] = (sum(f1_aft[l*0:l*1]) - sum(f1_10m[l*0:l*1]))**2  # Difference between after and 10min after
    sub_sq[3] = (sum(f1_aft[l*1:l*2]) - sum(f1_10m[l*1:l*2]))**2
    return sub_sq[:]

# Square Difference for first half of Step B
def get_sq_diff_B1(f1_bef,f1_aft, f1_10m,l):
    sub_sq = [0,0,0,0]
    sub_sq[0] = (sum(f1_bef[l*0:l*1]) - sum(f1_aft[l*0:l*1]))**2
    sub_sq[1] = (sum(f1_bef[l*1:l*2]) - sum(f1_aft[l*1:l*2]))**2
    sub_sq[2] = (sum(f1_aft[l*0:l*1]) - sum(f1_10m[l*0:l*1]))**2  # Difference between after and 10min after
    sub_sq[3] = (sum(f1_aft[l*1:l*2]) - sum(f1_10m[l*1:l*2]))**2
    return sub_sq[:]

# Square Difference for second half of Step B
def get_sq_diff_B2(f1_bef,f1_aft, f1_10m,l):
    sub_sq = [0,0,0,0]
    sub_sq[0] = (sum(f1_bef[l*2:l*3]) - sum(f1_aft[l*2:l*3]))**2
    sub_sq[1] = (sum(f1_bef[l*3:l*4]) - sum(f1_aft[l*3:l*4]))**2
    sub_sq[2] = (sum(f1_aft[l*2:l*3]) - sum(f1_10m[l*2:l*3]))**2  # Difference between after and 10min after
    sub_sq[3] = (sum(f1_aft[l*3:l*4]) - sum(f1_10m[l*3:l*4]))**2
    return sub_sq[:]

def f1(x, y, sub_sq):  # x:w1, y:w2
    return 1/2*sub_sq[0]*x**2 + 1/2*sub_sq[2]*x**2 + 1/2*sub_sq[0]*y**2 + 1/2*sub_sq[3]*y**2

# ∇f = [∂f/∂x, ∂f/∂y]^T
def gradient_f1(xy,sub_sq):
    x = xy[0]
    y = xy[1]
    # (Partial Difference of x, Partial Difference of y)
    return np.array([sub_sq[0]*x + sub_sq[2]*x, sub_sq[1]*y + sub_sq[3]*y]);

# Gradient Descent
def gradient_descent_method(gradient_f, init_pos, learning_rate, sub_sq):
    eps = 1e-10

    # 計算しやすいようnumpyのarrayとする
    init_pos = np.array(init_pos)
    pos = init_pos
    pos_history = [init_pos]
    iteration_max = 10000

    # 収束するか最大試行回数に達するまで
    for i in range(iteration_max):

        # 最急上昇法の場合は-を+にする
        #pos_new = pos - learning_rate * gradient_f(pos, sub_sq)
        pos_new = pos + learning_rate * gradient_f(pos, sub_sq)

        # If x or y is negative, set to zero (projected gradient descent)
        if (pos_new[0]+pos_new[1]) >= 1.0:
            break
        if pos_new[0] < 0.1:
            pos_new[0] = 0.1
        if pos_new[1] < 0.1:
            pos_new[1] = 0.1
        if 0.9 < pos_new[1]:
            pos_new[0] = 0.9
        if 0.9 < pos_new[1]:
            pos_new[1] = 0.9

        # 収束条件を満たせば終了
        # np.linalg.norm(): ユークリッド距離を計算する関数
        if abs(np.linalg.norm(pos - pos_new)) < eps:
            break

        pos = pos_new
        pos_history.append(pos)

        print("[{:3d}] i = {}, f(i) = {:7f}".format(i, pos_new, f1(pos_new[0],pos_new[1],sub_sq)))
    return (pos, np.array(pos_history))


# Draw raw signal
def draw_raw(fig,f1_bef,f1_aft, fig_10,d):
    ax0 = fig.add_subplot(111)
    c0 = ax0.plot(d, f1_bef[:], label="Before")
    c1 = ax0.plot(d, f1_aft[:], label="After")
    c1_ = ax0.plot(d, fig_10[:], label="10min After")
    ax0.set_title('Raw signal', fontsize=16)
    ax0.set_xlabel("Depth", fontsize=16)
    ax0.set_ylabel("Intensity", fontsize=16)
    ax0.legend()
    c2 = ax0.plot([4,4],[0,9], "black", linestyle="solid")
    c2 = ax0.plot([2.5,2.5],[0,9], "black", linestyle="dashed")
    c2 = ax0.plot([5.5,5.5],[0,9], "black", linestyle="dashed")

def draw_3D(ax,sub_sq):
    n = 20
    x = np.linspace(0, 1.0, n)
    y = np.linspace(0, 1.0, n)
    X, Y = np.meshgrid(x, y)
    Z = f1(X, Y, sub_sq)

    # Draw in 3D
    #ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0)
    ax.plot_wireframe(X, Y, Z, color="blue", zorder=1)
    ax.set_xlabel("w1", fontsize=12)
    ax.set_ylabel("w2", fontsize=12)
    ax.set_zlabel("Cost Function (Difference \n between before and after water)", fontsize=10)

    #ax.scatter(X, Y, Z, s=2, color="red", zorder=2)


def draw_contour(ax, sub_sq):
    # 等高線を描く
    n = 256
    x = np.linspace(-1.5, 1.5, n)
    y = np.linspace(-1.5, 1.5, n)
    X, Y = np.meshgrid(x, y)

    level_num = 20
    # 等高線で同じ高さとなるエリアを色分け
    ax.contourf(X, Y, f1(X, Y, sub_sq), level_num, alpha=.75, cmap=plt.cm.hot)
    # 等高線を引く
    C = ax.contour(X, Y, f1(X, Y, sub_sq), level_num, colors='black', linewidth=.5)
    ax.clabel(C, inline=1, fontsize=10)
    ax.set_title('contour')
    ax.set_xlabel("w1", fontsize=12)
    ax.set_ylabel("w2", fontsize=12)


def is_valid(num):
    return 0 < num < 1;


def main():
    d = np.linspace(1, 7, 60)
    #Define
    f1_bef = -10.429 + 26.44*d - 12.917*d**2 + 2.2929*d**3 - 0.1364*d**4
    f1_aft = -0.1429 + 2.3225*d + 0.9167*d**2 - 0.4394*d**3 + 0.0379*d**4
    f1_10m = -0.1326*d**4+2.0379*d**3-10.292*d**2+19.109*d-6.8571

    ######################## Step A optimization (separate to two segments) ########################
    l_2 = int(np.size(d,0)/2)
    sub_sq = get_sq_diff_A(f1_bef, f1_aft, f1_10m, l_2)

    fig0 = plt.figure(0)
    draw_raw(fig0, f1_bef, f1_aft, f1_10m, d)

    learning_rates = [0.000001]

    # 収束する様子を表示するためのグラフ
    fig2 = plt.figure(figsize=(8,6))

    for i, learning_rate in enumerate(learning_rates):
        ans, pos_history = gradient_descent_method(gradient_f1, (0.1, 0.1), learning_rate, sub_sq)

        ### 2D Plot ###
        # subplotの場所を指定
        ax = plt.subplot(111)  # 2行2列の意味
        # 等高線を描画
        draw_contour(ax, sub_sq)
        # グラフのタイトル
        ax.set_title("learning rate: " + str(learning_rate) + ", iteration: " + str(len(pos_history)))
        # 移動した点を表示
        for pos in pos_history:
            if is_valid(pos[0]) and is_valid(pos[1]):
                ax.plot(pos[0], pos[1], 'o')
        # 点同士を線で結ぶ
        for i in range(len(pos_history) - 1):
            x1 = pos_history[i][0]
            y1 = pos_history[i][1]
            x2 = pos_history[i + 1][0]
            y2 = pos_history[i + 1][1]
            if all([is_valid(v) for v in [x1, y1, x2, y2]]):
                ax.plot([x1, x2], [y1, y2], linestyle='-', linewidth=2)

        ### 3D Plot ###
        fig3d = plt.figure(2)
        ax2 = fig3d.add_subplot(111, projection='3d')
        draw_3D(ax2, sub_sq)
        #ax2.set_title("3D")
        for i in range(len(pos_history)):
            ax2.scatter(pos_history[i][0],pos_history[i][1],f1(pos_history[i][0],pos_history[i][1],sub_sq), s = 5, color = "red", zorder = 1)

    # タイトルが重ならないようにする
    fig2.tight_layout()

    # 画像を表示
    plt.show()

    # 画像を保存
    fig0.savefig('raw_signal.png')
    fig2.savefig('2d-result.png')

    w1 = pos[0]
    w2 = pos[1]
    #print("x = {:7f}, y = {:7f}, x+y = {:7f}".format(w1, w2, w1+w2))

    ######################## Step B optimization (separate to two more segments) ########################
    l_4 = int(np.size(d,0)/4)
    sub_sq_B1 = get_sq_diff_B1(f1_bef, f1_aft, f1_10m, l_4)  # squared diff of segment 1 (first segment of step1)
    sub_sq_B2 = get_sq_diff_B2(f1_bef, f1_aft, f1_10m, l_4)  # squared diff of segment 2 (second segment of step1)
    fig4 = plt.figure(figsize=(14, 6)) # Graph to draw convergence

    for i, learning_rate in enumerate(learning_rates):
        ans, pos_history_B1 = gradient_descent_method(gradient_f1, (0.1, 0.1), learning_rate, sub_sq_B1)
        ans, pos_history_B2 = gradient_descent_method(gradient_f1, (0.1, 0.1), learning_rate, sub_sq_B2)

        ### 2D Plot ###
        # Draw pos_history_B1
        ax = plt.subplot(1, 2, 1)  # Plot pos_history_B1
        draw_contour(ax, sub_sq_B1)
        ax.set_title("Segment1, learning rate: " + str(learning_rate) + ", iteration: " + str(len(pos_history_B1)))
        # Show moved points
        for posB1 in pos_history_B1:
            if is_valid(posB1[0]) and is_valid(posB1[1]):
                ax.plot(posB1[0], posB1[1], 'o')
        # Connect points with line
        for i in range(len(pos_history_B1) - 1):
            x1 = pos_history_B1[i][0]
            y1 = pos_history_B1[i][1]
            x2 = pos_history_B1[i + 1][0]
            y2 = pos_history_B1[i + 1][1]
            if all([is_valid(v) for v in [x1, y1, x2, y2]]):
                ax.plot([x1, x2], [y1, y2], linestyle='-', linewidth=2)


        # Draw pos_history_B2
        ax = plt.subplot(1, 2, 2)  # Plot pos_history_B2
        draw_contour(ax, sub_sq_B2)
        ax.set_title("Segment2, learning rate: " + str(learning_rate) + ", iteration: " + str(len(pos_history_B2)))
        # Show moved points
        for posB2 in pos_history_B2:
            if is_valid(posB2[0]) and is_valid(posB2[1]):
                ax.plot(posB2[0], posB2[1], 'o')
        # Connect points with line
        for i in range(len(pos_history_B2) - 1):
            x1 = pos_history_B2[i][0]
            y1 = pos_history_B2[i][1]
            x2 = pos_history_B2[i + 1][0]
            y2 = pos_history_B2[i + 1][1]
            if all([is_valid(v) for v in [x1, y1, x2, y2]]):
                ax.plot([x1, x2], [y1, y2], linestyle='-', linewidth=2)

        ### 3D Plot ###
        fig3d = plt.figure(figsize=(14, 6))
        ax2 = fig3d.add_subplot(121, projection='3d')
        draw_3D(ax2, sub_sq_B1)
        for i in range(len(pos_history_B1)):
            ax2.scatter(pos_history_B1[i][0],pos_history_B1[i][1],f1(pos_history_B1[i][0],pos_history_B1[i][1],sub_sq_B1), s = 5, color = "red", zorder = 1)

        ax2_ = fig3d.add_subplot(122, projection='3d')
        draw_3D(ax2_, sub_sq_B2)
        for i in range(len(pos_history_B2)):
            ax2_.scatter(pos_history_B2[i][0],pos_history_B2[i][1],f1(pos_history_B2[i][0],pos_history_B2[i][1],sub_sq_B2), s = 5, color = "red", zorder = 1)

    # タイトルが重ならないようにする
    fig4.tight_layout()

    # 画像を表示
    plt.show()

    # 画像を保存
    fig4.savefig('2d-result_step2.png')

    w1_1 = posB1[0] * w1
    w1_2 = posB1[1] * w1
    w2_1 = posB2[0] * w2
    w2_2 = posB2[1] * w2
    print("w1={:.4f}, w2={:.4f}, w1_1={:.4f}, w1_2={:.4f}, w2_1={:.4f}, w2_2={:.4f}, total={:.4f}"
          .format(w1, w2, w1_1, w1_2, w2_1, w2_2, w1_1+w1_2+w2_1+w2_2))


main()