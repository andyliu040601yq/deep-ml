def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	ans = []
	if len(a[0])!=len(b):
		return -1
	else:
		for row in a:
			temp = 0
			for j in range(len(b)):
				temp+=row[j]*b[j]
			ans.append(temp)
		return ans
				