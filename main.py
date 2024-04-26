from sympy.abc import x, y, z
from sympy import cos, tan, sin, exp, pprint
import math


def RungeKutta_method(f, y0, z0, Xs, h, rounding):
    def K1f(zk, h):
        return h*zk

    def H1f(xk, yk, zk, f, h):
        return h*f.subs([(x, xk), (y, yk), (z, zk)])

    def K2f(zk, h, H1):
        return h*(zk+(H1/3))

    def H2f(xk, yk, zk, f, h, K1, H1):
        return h*f.subs([(x, xk+h/3), (y, yk+K1/3), (z, zk+H1/3)])

    def K3f(zk, h, H2):
        return h*(zk+((2*H2)/3))

    def H3f(xk, yk, zk, f, h, K2, H2):
        return h*f.subs([(x, xk+((2*h)/3)), (y, yk+((2*K2)/3)), (z, zk+((2*H2)/3))])

    Ys = [y0]
    Zs = [z0]
    Ks = []
    Hs = []
    for i, xk in enumerate(Xs):
        Ks.append([])
        Hs.append([])
        print(f"{i}:")

        K1 = round(K1f(Zs[i], h), rounding)
        print(f"    K1: {K1}")
        Ks[i].append(K1)

        H1 = round(H1f(xk, Ys[i], Zs[i], f, h), rounding)
        print(f"    H1: {H1}")
        Hs[i].append(H1)

        K2 = round(K2f(Zs[i], h, H1), rounding)
        print(f"    K2: {K2}")
        Ks[i].append(K2)

        H2 = round(H2f(xk, Ys[i], Zs[i], f, h, K1, H1), rounding)
        print(f"    H2: {H2}")
        Hs[i].append(H2)

        K3 = round(K3f(Zs[i], h, H2), rounding)
        print(f"    K3: {K3}")
        Ks[i].append(K3)

        H3 = round(H3f(xk, Ys[i], Zs[i], f, h, K2, H2), rounding)
        print(f"    H3: {H3}")
        Hs[i].append(H3)

        dy = round(1/4*(K1+3*K3), rounding)
        print(f"    dy: {dy}")
        dz = round(1/4*(H1+3*H3), rounding)
        print(f"    dz: {dz}")

        ykN = round(Ys[i]+dy, rounding)
        print(f"    y{i+1} = {ykN}")
        zkN = round(Zs[i]+dz, rounding)
        print(f"    z{i+1} = {zkN}")

        Ys.append(ykN)
        Zs.append(zkN)

    return Ys


def EilerKoshi_method(f, y0, z0, Xs, h, rounding):

    def y_addition(yk, zk, h):
        return yk+h*zk

    def z_addition(xk, yk, zk, h, f):
        return zk + h*f.subs([(x, xk), (y, yk), (z, zk)])

    def z_final(xk, yk, zk, xkN, ykN, h, f):
        zA = z_addition(xk, yk, zk, h, f)
        return zk + (h/2)*(f.subs([(x, xk), (y, yk), (z, zk)])+f.subs([(x, xkN), (y, ykN), (z, zA)]))

    def y_final(xk, yk, zk, xkN, h, f):
        yA = y_addition(yk, zk, h)
        zF = z_final(xk, yk, zk, xkN, yA, h, f)
        return yk + (h/2)*(zk+zF)

    Ys = [y0]
    Zs = [z0]

    for i, xk in enumerate(Xs):
        if i == len(Xs)-1:
            break
        yk = round(y_final(xk, Ys[i], Zs[i], Xs[i+1], h, f), rounding)
        Ys.append(yk)
        Zs.append(
            round(z_final(xk, Ys[i], Zs[i], Xs[i+1], yk, h, f), rounding))

    return Ys


def Eiler_method(f, y0, z0, Xs, h, rounding):

    def y_next(yk, zk, h):
        return yk + zk * h

    def z_next(xk, yk, zk, h, f):
        return zk + h * f.subs([(x, xk), (y, yk), (z, zk)])

    Ys = [y0]
    Zs = [z0]

    for i, xk in enumerate(Xs):
        yk = Ys[i]
        zk = Zs[i]

        ykN = round(y_next(yk, zk, h), rounding)
        zkN = round(z_next(xk, yk, zk, h, f), rounding)

        Ys.append(ykN)
        Zs.append(zkN)

    return Ys


def errors(g, Xs, res, rounding):
    errors = []
    for i in range(len(Xs)):
        actRes = g.subs(x, Xs[i])
        errors.append(round(abs(actRes-res[i]), rounding))
    return errors


if __name__ == "__main__":
    a = 0
    b = 1
    y0 = 2
    z0 = 0
    f = y*(cos(x)**2)-z*tan(x)
    h = 0.1
    Xs = [x * h for x in range(a, b*10+1)]
    rounding = 5
    g = exp(sin(x)) + exp(-sin(x))

    RungeKut = RungeKutta_method(f, y0, z0, Xs, h, rounding)
    EilKosh = EilerKoshi_method(f, y0, z0, Xs, h, rounding)
    Eil = Eiler_method(f, y0, z0, Xs, h, rounding)

    RungeKut_Err = errors(g, Xs, RungeKut, rounding)
    EilKosh_Err = errors(g, Xs, EilKosh, rounding)
    Eil_Err = errors(g, Xs, Eil, rounding)

    print(f"Було дано диференційну функцію y'':")
    pprint(f)
    print(f"де z=y'\n\nМетодом Ейлера обчислили:\n{Eil}\nз похибками:\n{Eil_Err}\n\n")
    print(f"Методом Ейлера-Коші обчислили:\n{EilKosh}\nз похибками:\n{EilKosh_Err}\n\n")
    print(f"Методом Рунге-Кутта обчислили:\n{RungeKut}\nз похибками:\n{RungeKut_Err}\n\n")
