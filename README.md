<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
</head>
<body>
<h1>Project Documentation: Model Optimization</h1>
<p>This project includes scripts for optimizing the previously selected models from the "Model-selection-using-cross-validation" project by performing hyperparameter tuning. The scripts cover various aspects of the optimization process including feature selection, parameter tuning, and evaluation.</p>

<h2>Script A: Feature Selection for Model Optimization</h2>
<p><code>scripta.py</code> selects the top features from the dataset to be used for model optimization.</p>

<h3>Dependencies</h3>
<p>The script requires the following Python libraries:</p>
<ul>
<li>pandas</li>
<li>numpy</li>
<li>sklearn</li>
</ul>

<h3>Usage</h3>
<p>Ensure that the dataset is available as <code>reduceddataset.csv</code> in the working directory. Run the script using the command:</p>
<pre><code>python scripta.py</code></pre>

<h3>Functionality</h3>
<p>The script performs the following operations:</p>
<ol>
<li>Loads the dataset from a CSV file.</li>
<li>Transforms the label into a binary target variable.</li>
<li>Drops non-feature columns to prepare the feature set.</li>
<li>Applies SelectKBest for feature selection with chi-squared statistic.</li>
<li>Prints the selected features to be used for model optimization.</li>
</ol>

<h3>Expected Outputs</h3>
<p>The script will print out the list of selected features based on the chi-squared test.</p>

<h2>Script B: Hyperparameter Tuning with Randomized and Grid Search</h2>
<p><code>scriptb.py</code> performs hyperparameter tuning on a support vector machine classifier using both randomized and grid search methods.</p>

<h3>Dependencies</h3>
<p>The script requires the following Python libraries:</p>
<ul>
<li>pandas</li>
<li>numpy</li>
<li>sklearn</li>
<li>scipy.stats (for defining hyperparameter distributions)</li>
</ul>

<h3>Usage</h3>
<p>Ensure that the dataset is available as <code>reduceddataset.csv</code> in the working directory. Run the script using the command:</p>
<pre><code>python scriptb.py</code></pre>

<h3>Functionality</h3>
<p>The script performs the following operations:</p>
<ol>
<li>Loads the dataset and prepares the feature set and target variable.</li>
<li>Selects the best features using SelectKBest.</li>
<li>Defines a hyperparameter space with distributions such as loguniform.</li>
<li>Performs hyperparameter tuning on an SVM classifier using RandomizedSearchCV.</li>
<li>Performs an exhaustive hyperparameter search using GridSearchCV.</li>
<li>Prints the best hyperparameters and the performance of the tuned models.</li>
</ol>

<h3>Expected Outputs</h3>
<p>The script will output the best set of hyperparameters and the associated model performance metrics.</p>


<h2>Script C: Exhaustive Hyperparameter Tuning with Grid Search</h2>
<p><code>scriptc.py</code> focuses on exhaustive hyperparameter tuning of a support vector machine classifier using grid search.</p>

<h3>Dependencies</h3>
<p>The script requires the following Python libraries:</p>
<ul>
<li>pandas</li>
<li>numpy</li>
<li>sklearn</li>
<li>scipy.stats (for defining hyperparameter distributions)</li>
</ul>

<h3>Usage</h3>
<p>Ensure that the dataset is available as <code>reduceddataset.csv</code> in the working directory. Run the script using the command:</p>
<pre><code>python scriptc.py</code></pre>

<h3>Functionality</h3>
<p>The script performs the following operations:</p>
<ol>
<li>Loads the dataset and prepares the feature set and target variable.</li>
<li>Selects the best features using SelectKBest.</li>
<li>Defines a hyperparameter grid with parameters suitable for an SVM classifier.</li>
<li>Conducts an exhaustive search over the hyperparameter grid using GridSearchCV.</li>
<li>Outputs the best hyperparameters and evaluates the performance of the tuned model.</li>
</ol>

<h3>Expected Outputs</h3>
<p>The script will output the best hyperparameters found and performance metrics of the tuned model.</p>

<h2>Script D: Final Model Training and Serialization</h2>
<p><code>scriptd.py</code> performs the training of the support vector classifier with the selected features and saves the trained model to a file.</p>

<h3>Dependencies</h3>
<p>The script requires the following Python libraries:</p>
<ul>
<li>pandas</li>
<li>numpy</li>
<li>sklearn</li>
<li>pickle (for model serialization)</li>
</ul>

<h3>Usage</h3>
<p>Ensure that the dataset is available as <code>reduceddataset.csv</code> in the working directory. Run the script using the command:</p>
<pre><code>python scriptd.py</code></pre>

<h3>Functionality</h3>
<p>The script performs the following operations:</p>
<ol>
<li>Loads the dataset and separates features and target.</li>
<li>Selects the top features for the classifier using SelectKBest.</li>
<li>Trains a support vector classifier with optimal hyperparameters.</li>
<li>Saves the trained model to a pickle file for later use.</li>
</ol>

<h3>Expected Outputs</h3>
<p>The script will create a file named <code>saved_detector.pkl</code> which contains the trained model.</p>

<h2>Script E: Model Performance Evaluation</h2>
<p><code>scripte.py</code> evaluates the performance of the machine learning model by generating a classification report and confusion matrix based on the predictions logged during live classification.</p>

<h3>Dependencies</h3>
<p>The script requires the following Python libraries:</p>
<ul>
<li>pandas</li>
<li>numpy</li>
<li>json (for parsing log files)</li>
<li>sklearn (for computing metrics)</li>
<li>matplotlib (for visualization, if plotting is included)</li>
</ul>

<h3>Usage</h3>
<p>Ensure that the sample data is available as <code>samples.csv</code> and the log file <code>detector.log</code> is present. Run the script using the command:</p>
<pre><code>python scripte.py</code></pre>

<h3>Functionality</h3>
<p>The script performs the following operations:</p>
<ol>
<li>Loads the sample data and encodes the labels.</li>
<li>Reads and processes prediction logs from a file.</li>
<li>Constructs true label and prediction vectors.</li>
<li>Generates a classification report and confusion matrix.</li>
</ol>

<h3>Expected Outputs</h3>
<p>The script will print the classification report and confusion matrix to the console, providing insights into the classifier's performance.</p>
</body>
</html>
