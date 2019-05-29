from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

def get_sq_diff(f1_bef,f1_aft,l):
    sub_sq = [0,0,0]
    sub_sq[0] = (sum(f1_bef[l*0:l*1]) - sum(f1_aft[l*0:l*1]))**2
    sub_sq[1] = (sum(f1_bef[l*1:l*2]) - sum(f1_aft[l*1:l*2]))**2
    sub_sq[2] = (sum(f1_bef[l*2:l*3]) - sum(f1_aft[l*2:l*3]))**2
    return sub_sq[:]

def f1(x, y, sub_sq):  # x:f_bef, y:f_aft
    return 1/3*sub_sq[0]*x**2 + 1/3*sub_sq[1]*y**2 + 1/3*sub_sq[2]*(1-x-y)**2

# ∇f = [∂f/∂x, ∂f/∂y]^T
def gradient_f1(xy,sub_sq):
    x = xy[0]
    y = xy[1]
    # (Partial Difference of x, Partial Difference of y)
    return np.array([2/3*sub_sq[0]*x - 2/3*np.sqrt(sub_sq[2])*(1-x-y), 2/3*sub_sq[1]*y - 2/3*np.sqrt(sub_sq[2])*(1-x-y)]);

# Gradient Descent
# init_pos = 初期位置. e.g. (x, y)
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
        if pos_new[0] < 0.2:
            pos_new[0] = 0.2
        if pos_new[1] < 0.2:
            pos_new[1] = 0.2
        if 0.8 < pos_new[1]:
            pos_new[0] = 0.8
        if 0.8 < pos_new[1]:
            pos_new[1] = 0.8

        # 収束条件を満たせば終了
        # np.linalg.norm(): ユークリッド距離を計算する関数
        if abs(np.linalg.norm(pos - pos_new)) < eps:
            break

        pos = pos_new
        pos_history.append(pos)

        print("[{:3d}] i = {}, f(i) = {:7f}".format(i, pos_new, f1(pos_new[0],pos_new[1],sub_sq)))
    return (pos, np.array(pos_history))


# Draw raw signal
def draw_raw(fig,f1_bef,f1_aft,d):
    ax0 = fig.add_subplot(111)
    c0 = ax0.plot(d, f1_bef[:], label="Before")
    c1 = ax0.plot(d, f1_aft[:], label="After")
    ax0.set_title('Raw signal', fontsize=16)
    ax0.set_xlabel("Depth", fontsize=16)
    ax0.set_ylabel("Intensity", fontsize=16)
    ax0.legend()
    c2 = ax0.plot([3,3],[0,9], "black", linestyle="dashed")
    c3 = ax0.plot([5,5],[0,9], "black", linestyle="dashed")


# 値の高低を色で表現
def draw_color_map(fig,sub_sq):
    ax1 = fig.add_subplot(111)
    n = 20
    x = np.linspace(0, 1.0, n)
    y = np.linspace(0, 1.0, n)
    X, Y = np.meshgrid(x, y)
    Z = f1(X, Y, sub_sq)

    pc = ax1.pcolor(X, Y, Z, cmap='RdBu')

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])  # カラーバー用のaxesを用意
    fig.colorbar(pc, cax=cbar_ax)  # カラーバーを描画


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


def is_valid(num):
    return 0 < num < 1;


def main():
    d = np.linspace(1, 7, 60)
    #Define
    f1_bef = -10.429 + 26.44*d - 12.917*d**2 + 2.2929*d**3 - 0.1364*d**4
    f1_aft = -0.1429 + 2.3225*d + 0.9167*d**2 - 0.4394*d**3 + 0.0379*d**4

    l = int(np.size(d,0)/3)
    sub_sq = get_sq_diff(f1_bef, f1_aft, l)

    fig0 = plt.figure(0)
    draw_raw(fig0, f1_bef, f1_aft, d)

    # カラーマップを表示
    fig1 = plt.figure(1)
    draw_color_map(fig1, sub_sq)

    learning_rates = [0.000001]

    # 収束する様子を表示するためのグラフ
    fig2 = plt.figure(3)

    for i, learning_rate in enumerate(learning_rates):
        ans, pos_history = gradient_descent_method(gradient_f1, (0.1, 0.1), learning_rate, sub_sq)

        ### 2D Plot ###
        # subplotの場所を指定
        ax = plt.subplot(2, 2, (i + 1))  # 2行2列の意味
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
    fig1.savefig('color_map.png')
    fig2.savefig('2d-result.png')

    w1 = pos[0]
    w2 = pos[1]
    w3 = 1 - w1 - w2
    print("x = {:7f}, y = {:7f}, z = {:7f}".format(w1, w2, w3))

main()