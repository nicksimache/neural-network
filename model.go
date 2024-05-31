package main

import (
	"math"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"
)

type NeuralNetwork struct {
	layers  int
	sizes   []int
	biases  []*mat.Dense
	weights []*mat.Dense
}

type Tuple[U any, V any] struct {
	first  U
	second V
}

func newNetwork(sizes []int) *NeuralNetwork {

	randSource := rand.NewSource(time.Now().UnixNano())
	random := rand.New(randSource)

	weights := make([]*mat.Dense, len(sizes)-1)
	biases := make([]*mat.Dense, len(sizes)-1)

	for i := 0; i < len(sizes)-1; i++ {

		//weights
		rowsW := sizes[i+1]
		colsW := sizes[i]
		dataW := make([]float64, rowsW*colsW)
		for j := range dataW {
			dataW[j] = random.Float64()
		}

		//biases
		rowsB := sizes[i+1]
		dataB := make([]float64, rowsB)
		for j := range dataB {
			dataB[j] = rand.NormFloat64()
		}

		weights[i] = mat.NewDense(rowsW, colsW, dataW)
		biases[i] = mat.NewDense(rowsB, 1, dataB)
	}

	return &NeuralNetwork{
		layers:  len(sizes),
		sizes:   sizes,
		weights: weights,
		biases:  biases,
	}
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func sigmoidPrime(x float64) float64 {
	return sigmoid(x) * (1.0 - sigmoid(x))
}

func (nn *NeuralNetwork) Feedforward(a *mat.Dense) *mat.Dense {

	for i := range nn.biases {

		z := mat.NewDense(nn.biases[i].RawMatrix().Rows, 1, nil)
		z.Product(nn.weights[i], a)

		z.Add(z, nn.biases[i])

		// applying sigmoid func
		rows, _ := z.Dims()
		data := make([]float64, rows)
		for i := range data {
			data[i] = sigmoid(z.At(i, 0))
		}

		a = mat.NewDense(rows, 1, data)

	}

	return a
}

func (nn *NeuralNetwork) SGD(trainingData []Tuple[float64, float64], epochs int, eta float64, miniBatchSize int) {
	n := len(trainingData)

	for j := 0; j < epochs; j++ {
		rand.Shuffle(len(trainingData), func(i, j int) {
			trainingData[i], trainingData[j] = trainingData[j], trainingData[i]
		})

		var miniBatches [][]Tuple[float64, float64]

		for k := 0; k < n; k += miniBatchSize {
			miniBatch := trainingData[k : k+miniBatchSize]
			miniBatches = append(miniBatches, miniBatch)

		}

		for _, miniBatch := range miniBatches {
			nn.updateMiniBatch(miniBatch, eta)
		}
	}

}

func (nn *NeuralNetwork) updateMiniBatch(miniBatch []Tuple[float64, float64], eta float64) {
	nablaB := make([]*mat.Dense, len(nn.biases))
	for i, b := range nn.biases {
		r, c := b.Dims()
		nablaB[i] = mat.NewDense(r, c, nil)
	}

	nablaW := make([]*mat.Dense, len(nn.weights))
	for i, w := range nn.weights {
		r, c := w.Dims()
		nablaW[i] = mat.NewDense(r, c, nil)
	}

}
