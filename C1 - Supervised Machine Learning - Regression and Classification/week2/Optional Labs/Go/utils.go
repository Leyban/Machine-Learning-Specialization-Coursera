package main

import (
	"encoding/csv"
	"log"
	"math"
	"os"
	"strconv"
)

func getMean(a []float64) float64 {
	sum := 0.
	for i := range a {
		sum += a[i]
	}

	return sum / float64(len(a))
}

func getStd(a []float64) float64 {
	s := 0.

	m := getMean(a)

	for i := range a {
		s += math.Pow((a[i] - m), 2)
	}

	return math.Sqrt(s / float64(len(a)))
}

func readCsvFile(filePath string) [][]string {
	f, err := os.Open(filePath)
	if err != nil {
		log.Fatal("Unable to read input file "+filePath, err)
	}
	defer f.Close()

	csvReader := csv.NewReader(f)
	records, err := csvReader.ReadAll()
	if err != nil {
		log.Fatal("Unable to parse file as CSV for "+filePath, err)
	}

	return records
}

func readFloatsCsv(filePath string) [][]float64 {
	records := readCsvFile(filePath)

	recordsf := make([][]float64, len(records))

	for i := range records {
		recordsf[i] = make([]float64, len(records[i]))
		for j := range records[i] {
			f, err := strconv.ParseFloat(records[i][j], 64)
			if err != nil {
				log.Fatal("Unable to parse float: (", i, ",", j, ")", err)
			}
			recordsf[i][j] = f
		}
	}

	return recordsf
}

func getXandY(data [][]float64, y_col int) (X [][]float64, y []float64) {

	X = make([][]float64, len(data))
	y = make([]float64, len(data))

	for i := range data {
		X[i] = make([]float64, len(data[i])-1)
		for j := range data[i] {
			if j == y_col {
				y[i] = data[i][j]
			} else {
				X[i][j] = data[i][j]
			}
		}
	}

	return X, y
}
