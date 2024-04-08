package main

import "fmt"

func main() {
	// lab_02()

	data := readFloatsCsv("../data/houses.txt")

	X, y := getXandY(data, 4)

	w_init := make([]float64, len(X[0]))
	b_init := 0.

	alpha := 1.e-1
	iterations := 1000

	X_norm := zScoreNormalization(X)

	w_final, b_final, _ := batchGradientDescent(X_norm, y, w_init, b_init, computeLoss, computeGradient, alpha, int64(iterations))

	fmt.Println(w_final, b_final)

}
