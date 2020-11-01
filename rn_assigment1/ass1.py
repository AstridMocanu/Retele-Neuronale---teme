import numpy as np
def citire(A, B):
    count = -1
    A=[[0 for i in range(3)] for j in range(3)]

    with open("matrice.txt") as file:
        for line in file:
            print(line)
            line=line.replace(" ","")
            # parsare line
            count += 1
            # elim =
            line, b = line.split("=")
            B.append(float(b))
            # elim *
            line.replace("*", "")
            # obtinere coef x,y,z
            for char in ["x", "y", "z"]:
                if char in line:
                    if line[:line.index(char)] == "-":
                        A[count][ord(char) - ord("x")] = -1
                    elif line[:line.index(char)] == "+":
                        A[count][ord(char) - ord("x")] = 1
                    else:
                        A[count][ord(char) - ord("x")] = int(line[:line.index(char)])
                    line=line[line.index(char)+1:]
                else:
                    A[count][ord(char) - ord("x")] = 0
    print(A,B)
    return A,B


def det(A):
    return A[0][0] * A[1][1] * A[2][2] + A[1][0] * A[0][2] * A[2][1] + A[0][1] * A[1][2] * A[2][0] - A[0][2] * A[1][1] * \
           A[2][0] - A[0][0] * A[2][1] * A[1][2] - A[2][2] * A[0][1] * A[1][0]


def trans(A):
    a = [[], [], []]
    for i in range(0, 3):
        for j in range(0, 3):
            a[i][j] = A[j][i]
    return a


"""    
123      147
456 =>   258 a[0][1]=a[1][0]
789      369
"""


def adj(A):
    A_star = []
    for i in range(0, 3):
        for j in range(0, 3):
            mat = []
            for k in range(0, 3):
                for p in range(0, 3):
                    if k != i and p != j:
                        mat.append(A[k][p])
            A_star.append((-1) ** (i + j) * (mat[0] * mat[3] - mat[1] * mat[2]))
    return A_star


def inv(A):
    A_star = adj(A)
    print("adj")
    print(A_star)
    new_A = [[],[],[]]
    for i in range(0, 3):
        new_A[i]=[A_star[3*i], A_star[3*i + 1], A_star[3*i + 2]]
    d=det(A)
    if det(A) != 0:
        return [[new_A[i][j]/d for i in range(0,3)] for j in range(3)]


    else:
        return None


def solutie(A, B):
    X = []
    print(A,B)
    A=inv(A)
    for i in range(0, 3):
        X.append(0)
        for j in range(0, 3):
            X[i] += A[i][j] * B[j]
    return X


def rez_sistem():
    B = []
    A = [[], [], []]
    A,B = citire(A, B)
    x = solutie(A, B)
    return x


def rez_sistem_numpy():
    B = []
    A = [[], [], []]
    A, B = citire(A, B)
    a=np.array(A)
    b=np.array(B)

    d=np.linalg.det(a)
    print(d)
    if(d!=0):
        X=np.linalg.inv(a)@b
    else:
        X=None
    return X



print(rez_sistem())
print(rez_sistem_numpy())
