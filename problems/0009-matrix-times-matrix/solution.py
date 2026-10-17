def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:
    if len(a[0]) != len(b):
        return -1

    ans = []

    for i in range(len(a)):
        temp = []
        for j in range(len(b[0])):
            num = 0
            for k in range(len(a[0])):
                num+=a[i][k]*b[k][j]
            temp.append(num)
        ans.append(temp)
    return ans