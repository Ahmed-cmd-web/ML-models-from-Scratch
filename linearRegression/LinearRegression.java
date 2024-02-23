package linearRegression;


/**
 * LinearRegression
 */
public class LinearRegression {

    private double[] weights;
    public double[] getWeights() {
        return weights;
    }

    private double bias;
    private double learningRate = 0.001;
    private int numberOfIterations = 100000;

    public LinearRegression(double learningRate, int numberOfIterations) {
        this.learningRate = learningRate;
        this.numberOfIterations = numberOfIterations;
    }

    public LinearRegression(int numberOfIterations) {
        this.bias = 0;
        this.numberOfIterations = numberOfIterations;
    }

    public LinearRegression() {
        this.bias = 0;
    }
    public LinearRegression(double learningRate) {
        this.bias = 0;
        this.learningRate = learningRate;
    }


    public void fit(double[][] X, double[] Y) {
        int numOfSamples = X.length;
        int numOfFeats = numOfSamples > 0 ? X[0].length : 0;
        this.weights = new double[numOfFeats];
        for (int j = 0; j < this.numberOfIterations; j++) {
            // get Wx for each point
            double[] y_preds = multiplyEach(X, this.weights);

            // add bias to each point
            y_preds = sum(y_preds, this.bias);

            //  y^ - y
            double[] predsMinusActual = new double[y_preds.length];

            for (int i = 0; i < predsMinusActual.length; i++)
                predsMinusActual[i] = y_preds[i]-Y[i];

            double dw = 0;
            for (int i = 0; i < predsMinusActual.length; i++)
                dw += X[i][0] * predsMinusActual[i];


            for (int i = 0; i < this.weights.length; i++)
                this.weights[i] -= this.learningRate * dw / (numOfSamples * 1.0);
            this.bias -= this.learningRate * sum(predsMinusActual) / (numOfSamples * 1.0);

        }

    }




    public double predict(double[] X) {
        return dot(this.weights, X) + this.bias;
    }






    private double[] multiplyEach(double[][] a1, double[] a2) {
        double[] res = new double[a1.length];

        for (int i = 0; i < a1.length; i++) {
            double[] tmp = new double[a1[i].length];
            for (int j = 0; j < a1[i].length; j++)
                tmp[j] = a1[i][j] * a2[j];
            res[i] = sum(tmp);
        }
        return res;

    }

    private double dot(double[] a1, double[] a2) {
        double res = 0;

        for (int i = 0; i < a2.length; i++)
            res += a1[i] * a2[i];

        return res;

    }

    private double sum(double[] arr) {
        double sum = 0;
        for (double d : arr)
            sum += d;
        return sum;
    }

    private double[] sum(double[] arr, double x) {
        double[] res = new double[arr.length];
        for (int i = 0; i < res.length; i++)
            res[i] =arr[i]+ x;

        return res;

    }

}