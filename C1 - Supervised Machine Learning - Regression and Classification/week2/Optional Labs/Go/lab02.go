package main

import (
	"fmt"
	"time"
)

func lab_02() {
	X_train := [][]float64{
		{2104, 5, 1, 45},
		{1416, 3, 2, 40},
		{852, 2, 1, 35},
	}

	y_train := []float64{460, 232, 178}
	fmt.Println("Y_train", y_train)

	b_init := 785.1811367994083
	w_init := []float64{0.39133535, 18.75376741, -53.36032453, -26.42131618}

	x_vec := X_train[0]

	fmt.Println(predictSingleLoop(x_vec, w_init, b_init))

	fmt.Println(computeLoss(X_train, y_train, w_init, b_init))

	tmp_dj_dw, tmp_dj_db := computeGradient(X_train, y_train, w_init, b_init)

	fmt.Println("dj_db: ", tmp_dj_db)
	fmt.Println("dj_dw: ", tmp_dj_dw)

	initial_w := make([]float64, len(X_train[0]))
	initial_b := 0.

	iterations := 1000
	alpha := 5.0e-7

	timeStart := time.Now()
	w_final, b_final, _ := batchGradientDescent(X_train, y_train, initial_w, initial_b, computeLoss, computeGradient, alpha, int64(iterations))
	fmt.Println("time elapsed: ", time.Since(timeStart))

	fmt.Println("b,w", b_final, w_final)
	for i := range X_train {
		fmt.Println("Prediction: ", dotProduct(X_train[i], w_final)+b_final, "Target: ", y_train[i])
	}
}
