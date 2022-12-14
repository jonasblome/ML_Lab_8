package de.htw.ml;

import java.util.Arrays;

import org.jblas.FloatMatrix;

import de.htw.ml.data.CreditDataset;
import de.htw.ml.data.Dataset;

/**
 * There are some TODOs here.
 * Combines the different logistic regression modules into one system.
 * 
 * @author Nico Hezel
 */
public class ExpertSystem {

	protected LogisticRegression regression;
	
	protected int[] categories; 			// The system should be able to predict these labels
	protected FloatMatrix[] thetas; 		// Weights of a logistic regression model for a label
	protected float[][] predictionRates;	// The prediction rates during training
	protected float[][] trainErrors;		// The error rates during training

	
	public ExpertSystem(int trainingIterations, float learnRate, int[] categories) {
		this.regression = new LogisticRegression(trainingIterations, learnRate);
		this.thetas = new FloatMatrix[categories.length];
		this.predictionRates = new float[categories.length][];
		this.trainErrors = new float[categories.length][];		
		this.categories = categories;
	}
	
	/**
	 * Trains for every unique label a separate logistic regression model.
	 * 
	 * @param dataset
	 */
	public void train(CreditDataset dataset) {
		// Train a logistic regression for each category
		for (int i = 0; i < categories.length; i++) {	
			final int category = categories[i];
			
			// Create the training set for this category
			final Dataset subset = dataset.getSubset(category);
			final float ratio = (subset.getYTrain().sum() / subset.getYTrain().rows * 100);
			System.out.printf("Train category %d (%.2f%% share with %d elements)\n", category, ratio, subset.getYTrain().getRows());
			
			// Start the training process
			thetas[i] = regression.train(subset.getXTest(), subset.getYTest(), subset.getXTrain(), subset.getYTrain());
			predictionRates[i] = regression.getPredictionRates();
			trainErrors[i] = regression.getTrainError();
			System.out.printf("Best prediction rate %.2f%%\n\n", (new FloatMatrix(predictionRates[i])).max());
		}
	}

	public float test(Dataset dataset) {
		FloatMatrix xTest = dataset.getXTest();
		FloatMatrix yTest = dataset.getYTest();
		
		// The predictions for each label
		FloatMatrix[] hypothesisArr = Arrays.stream(thetas).map(theta -> LogisticRegression.predict(xTest, theta)).toArray(FloatMatrix[]::new);

		// Run through all predictions ...
		int numCorrect = 0;

		for (int r = 0; r < yTest.getRows(); r++) {
			int expectedLabel = (int)yTest.data[r];
			
			// ... and find the strongest one (the highest value)
			float hypothesisLabel = 0;
			float highestPrediction = 0;

			for(int h = 0; h < hypothesisArr.length; h++) {
				if(hypothesisArr[h].data[r] > highestPrediction) {
					highestPrediction = hypothesisArr[h].data[r];
					hypothesisLabel = categories[h];
				}
			}

			// count how many times the system found the right label
			if(expectedLabel == hypothesisLabel)
				numCorrect++;
		}

		return (float) numCorrect / yTest.getRows();
	}
	
	public float[][] getPredictionRates() {
		return predictionRates;
	}
	
	public float[][] getTrainErrors() {
		return trainErrors;
	}
}
