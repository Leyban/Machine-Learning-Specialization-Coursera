package main

import (
	"fmt"
	"time"
)

func lab05() {
	X_train := [][]float64{
		{0.5, 1.5},
		{1, 1},
		{1.5, 0.5},
		{3, 0.5},
		{2, 2},
		{1, 2.5},
	}

	y_train := []float64{0, 0, 0, 1, 1, 1}

	w_tmp := []float64{1, 1}
	b_tmp := -3

	fmt.Println(compute_loss_logistic(X_train, y_train, w_tmp, float64(b_tmp)))
}

func lab06() {
	X_train := [][]float64{
		{0.5, 1.5},
		{1, 1},
		{1.5, 0.5},
		{3, 0.5},
		{2, 2},
		{1, 2.5},
	}
	y_train := []float64{0, 0, 0, 1, 1, 1}

	w_tmp := []float64{2, 3}
	b_tmp := 1.

	dj_dw_tmp, dj_db_tmp := compute_gradient_logistic(X_train, y_train, w_tmp, b_tmp)

	fmt.Println("dj_db:", dj_db_tmp)
	fmt.Println("dj_dw:", dj_dw_tmp)

	w_init := make([]float64, len(X_train[0]))
	b_init := 0.
	alpha := 0.1
	iters := 10000

	start := time.Now()
	w_out, b_out, _ := gradient_descent(X_train, y_train, w_init, b_init, alpha, int64(iters))
	fmt.Println("Time Elapsed: ", time.Since(start))
	fmt.Println("Final params -- w: ", w_out, "b:", b_out)
}

func lab09() {
	X_tmp := [][]float64{
		{4.17022005e-01, 7.20324493e-01, 1.14374817e-04},
		{3.02332573e-01, 1.46755891e-01, 9.23385948e-02},
		{1.86260211e-01, 3.45560727e-01, 3.96767474e-01},
		{5.38816734e-01, 4.19194514e-01, 6.85219500e-01},
		{2.04452250e-01, 8.78117436e-01, 2.73875932e-02},
	}

	y_tmp := []float64{0, 1, 0, 1, 0}
	w_tmp := []float64{0.67046751, 0.4173048, 0.55868983}
	b_tmp := 0.5

	lambda_tmp := 0.7

	dj_dw, dj_db := compute_gradient_logistic_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

	fmt.Println(dj_dw)
	fmt.Println(dj_db)
}
