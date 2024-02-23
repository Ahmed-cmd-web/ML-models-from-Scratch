package linearRegression;

import java.util.Arrays;




public class Index {
    public static void main(String[] args) {
        var lr   = new LinearRegression();


        var data = Util.train_test_split(Util.XValues, Util.YValues);   // data[0]=XTrain,data[1]=YTrain,data[2]=XTest,data[3]=YTest
        lr.fit(data[0], Util.flatten(data[1]));


        double[] predictions = new double[data[2].length];
        for (int i = 0; i < predictions.length; i++)
            predictions[i] = lr.predict(data[2][i]);

        System.out.println(Arrays.toString(predictions));


    }
}
