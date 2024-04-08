package main

import "fmt"

func main() {
	zTmp := []float64{
		-10,
		-9,
		-8,
		-7,
		-6,
		-5,
		-4,
		-3,
		-2,
		-1,
		0,
		1,
		2,
		3,
		4,
		5,
		6,
		7,
		8,
		9,
		10,
	}

	y := make([]float64, len(zTmp))

	for i := range zTmp {
		y[i] = sigmoid(zTmp[i])
		fmt.Printf("%.2f, %v \n", y[i], zTmp[i])
	}
}
