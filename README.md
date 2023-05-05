Download Link: https://assignmentchef.com/product/solved-cse574-assignment1-linear-models-for-supervised-learning
<br>
<strong>Note </strong>A zipped file containing skeleton Python script files and data is provided. Note that for each problem, you need to write code in the specified function withing the Python script file. <strong>Do not use any Python libraries/toolboxes, built-in functions, or external tools/libraries that directly perform classification, regression, function fitting, etc.</strong>. Using any ML library calls, such as scikit-learn, will result in 0 points for the corresponding problem.

<strong>Evaluation </strong>We will evaluate your code by executing PA1Script.ipynb file, which will internally call the problem specific functions. Also submit an assignment report (pdf file) summarizing your findings. In the problem statements below, the portions under REPORT heading need to be discussed in the assignment report.

<strong>Data Sets         </strong>Two data sets are provided:

<ol>

 <li>A medical data set is provided in the file “diabetes.pickle” along with the target assignment. The input variables correspond to measurements (physical, physiological, and blood related) for a given patient and the target variable corresponds to the level of diabetic condition in the patient. It contains:

  <ul>

   <li><strong>x<sub>train </sub></strong>(242 × 64) and <strong>y<sub>train </sub></strong>(242 × 1) for training.</li>

   <li><strong>x<sub>test </sub></strong>(200 × 64) and <strong>y<sub>test </sub></strong>(200 × 1) for testing.</li>

  </ul></li>

 <li>A 2D sample data set in the file “sample.pickle”.</li>

</ol>

<strong>Submission         </strong>You are required to submit a single file called <em>pa1.zip </em>using UBLearns. File <em>pa1.zip </em>must contain 2 files: <em>report.pdf </em>and <em>PA1Script.ipynb</em>.

<ul>

 <li>Submit your report in a pdf format. Please indicate the <strong>team members</strong>, <strong>group number</strong>, and your <strong>course number </strong>on the top of the report.</li>

 <li>The code file should contain all implemented functions. Please do not change the name of the file.</li>

</ul>

<strong>Please make sure that your group is enrolled in the UBLearns system</strong>: You should submit one solution per group through the groups page. <em>If you want to change the group, contact the instructors.</em>

<strong>Project report: </strong>The hard-copy of report will be collected in class at due date. The problem descriptions specify what is needed in the report for each problem.

<h2>Part I – Linear Regression</h2>

In this part you will implement the direct and gradient descent based learning methods for Linear Regression and compare the results on the provided “diabetes” dataset.

<strong>Problem 1: Linear Regression with Direct Minimization </strong>

Implement <em>ordinary least squares </em>method to estimate regression parameters by minimizing the squared loss.




In matrix-vector notation, the loss function can be written as:

<strong>Xw</strong>)<sup>&gt;</sup>(<strong>y </strong>−<strong>Xw</strong>)

where <strong>X </strong>is the input data matrix, <strong>y </strong>is the target vector, and <strong>w </strong>is the weight vector for regression.

You need to implement the function learnOLERegression. Also implement the function testOLERegression to apply the learnt weights for prediction on both training and testing data and to calculate the <em>root mean squared error </em>(RMSE):




<strong>REPORT 1. </strong>

Calculate and report the RMSE for training and test data for two cases: first, without using an intercept (or bias) term, and second with using an intercept. Which one is better?

<strong>Problem 2: Using Gradient Descent for Linear Regression Learning </strong>

As discussed in class, regression parameters can be calculated directly using analytical expressions (as in Problem 1). However, to avoid computation of (<strong>X</strong><sup>&gt;</sup><strong>X</strong>)<sup>−1</sup>, another option is to use gradient descent to minimize the loss function. In this problem, you have to implement the gradient descent procedure for estimating the weights <strong>w</strong>, where the gradient is given by:

∇<em>J</em>(<strong>w</strong>) = <strong>X</strong><sup>&gt;</sup><strong>Xw </strong>−<strong>X</strong><sup>&gt;</sup><strong>y                                                                               </strong>

You need to use the minimize function (from the <strong>scipy </strong>library). You need to implement a function regressionObjVal to compute the squared error (See (2)) and a function regressionGradient to compute its gradient with respect to <strong>w</strong>. In the main script, this objective function and the gradient function will be used within the minimizer (See https://docs.scipy.org/doc/scipy/reference/generated/scipy. optimize.minimize.html for more details).

<strong>REPORT 2. </strong>

Using testOLERegression, calculate and report the RMSE for training and test data after gradient descent based learning. Compare with the RMSE after direct minimization. Which one is better?

<h2>Part II – Linear Classifiers</h2>

In this part you will implement three different linear classifiers using different optimization algorithms and compare the results on the provided data set. You will also have to draw the discrimination boundaries for the three classifiers and compare. The three classifiers are:

<ol>

 <li>Perceptron</li>

 <li>Logistic Regression</li>

 <li>Linear Support Vector Machine (SVM)</li>

</ol>

For each classifier, the decision rule is the same, i.e., the target, <em>y<sub>i</sub></em>, for a given input, <strong>x</strong><em><sub>i </sub></em>is given by:

−1       if <strong>w</strong><sup>&gt;</sup><strong>x</strong><em><sub>i </sub>&lt; </em>0

+1       if <strong>w</strong>&gt;<strong>x</strong><em>i </em>≥ 0

where <strong>w </strong>are the weights representing to the linear discriminating boundary. We will assume that we have included a constant term in <strong>x</strong><em><sub>i </sub></em>and a corresponding weight in <strong>w</strong>. While all three classifiers have the same decision function<sup>1</sup>

For this part, you will implement the training algorithms for the three different linear classifiers, learn a model for the sample training data and report the accuracy on the sample training and test data sets. The sample training and test data sets are included in the “sample.pickle” file.

<strong>Problem 3: Using Gradient Descent for Perceptron Learning </strong>

For this problem, you will training a perceptron, which has a squared loss function which is exactly the same as linear regression (See (1)), i.e.,




which means that you can call the same functions, regressionObjVal and regressionGradient, implemented in Problem 2, to train the perceptron. Implement two functions:

<ol>

 <li>a testing function, predictLinearModel that returns the predictions of a model on a test data set</li>

 <li>an evaluation function, evaluateLinearModel, that computes the accuracy of the model on the test data by calculating the fraction of observations for which the predicted label is same as the true label.</li>

</ol>

<strong>REPORT 3. </strong>

Train the perceptron model by calling the scipy.optimize.minimize method and use the evaluateLinearModel to calculate and report the accuracy for the training and test data.

<strong>Problem 4: Using Newton’s Method for Logistic Regression Learning </strong>

For this problem, you will train a logistic regression model, whose loss function (also known as the <em>logistic-loss </em>or <em>log-loss</em>) is given by:

))

<sup>1</sup>For Logistic Regression, typically a different formulation is presented. The decision rule is written as:

1         if <em>θ<sub>i </sub>&lt; </em>0<em>.</em>5

(6)

+1          if <em>θ<sub>i </sub></em>≥ 0<em>.</em>5

where,




However, one can see that it is equivalent to checking if <strong>w</strong><sup>&gt;</sup><strong>x</strong><em><sub>i </sub>&lt; </em>0 or not.

The gradient for this loss function is given by, as derived in the class:

<strong>x</strong><em><sub>i                                                                   </sub></em>

The <em>Hessian </em>for the loss function is given by:

<strong>H</strong>

<table width="623">

 <tbody>

  <tr>

   <td width="145"><strong>Newton’s Method</strong></td>

   <td width="478">The update rule is given by:<strong>w</strong>(<em>t</em>) = <strong>w</strong>(<em>t</em>−1) + <em>η</em><strong>H</strong>−1(<strong>w</strong>(<em>t</em>−1))∇<em>J</em>(<strong>w</strong>(<em>t</em>−1))</td>

  </tr>

 </tbody>

</table>

However, for this assignment we will be using the scipy.optimize.minimize function again, with method = ’Newton-CG’, for training using the Newton’s method. This will need you to implement the following three functions:

<ol>

 <li>logisticObjVal – compute the logistic loss for the given data set (See (9)).</li>

 <li>logisticGradient – compute the gradient vector of logistic loss for the given data set (See (10)).</li>

 <li>logisticHessian – compute the Hessian matrix of logistic loss for the given data set (See (11)).</li>

</ol>

<strong>REPORT 4. </strong>

Train the logistic regression model by calling the scipy.optimize.minimize method, and use the evaluateLinearModel to calculate and report the accuracy for the training and test data.

<strong>Problem 5: Using Stochastic Gradient Descent Method for Training Linear Support Vector Machine (20 code + 5 report = 25 points)</strong>

While we will study the quadratic optimization formulation for SVMs in class, we can also train the SVM directly using the <em>hinge-loss </em>given by:

<em>n</em>

<em>J</em>(<strong>w</strong>) = <sup>X</sup>max(0<em>,</em>1 − <em>y<sub>i</sub></em><strong>w</strong><sup>&gt;</sup><strong>x</strong><em><sub>i</sub></em>)

<em>i</em>=1

Clearly, the above function is not as easily differentiable as the <em>squared-loss </em>and <em>logistic-loss </em>functions above, we can devise a simple <em>Stochastic Gradient Descent </em>(SGD) based method for learning <strong>w</strong>. Note that, for a single observation, the loss is given by:

<em>J<sub>i</sub></em>(<strong>w</strong>)      =        max(0<em>,</em>1 − <em>y<sub>i</sub></em><strong>w</strong><sup>&gt;</sup><strong>x</strong><em><sub>i</sub></em>)

&gt;                                &gt;

(14)

Thus, the gradient of <em>J<sub>i</sub></em>(<strong>w</strong>) can be written as:




The training can be done using the following algorithm:

1: <strong>w </strong>← [0<em>,</em>0<em>,…,</em>0]<sup>&gt;</sup>

2: <strong>for </strong><em>t</em>=1<em>,</em>2<em>,…T </em><strong>do</strong>

3:                    <em>i </em>← <em>RandomSample</em>(1<em>…n</em>)

4:                    <strong>if </strong><em>y<sub>i</sub></em><strong>w</strong><sup>&gt;</sup><strong>x</strong><sup>(<em>i</em>) </sup><em>&lt; </em>1 <strong>then</strong>

5:                        <strong>w </strong>← <strong>w </strong>+ <em>ηy<sub>i</sub></em><strong>x</strong><em><sub>i</sub></em>

6:                <strong>end if </strong>7: <strong>end for</strong>

You have to implement a function trainSGDSVM that learns the optimal weight, <strong>w </strong>using the above algorithm.

<strong>REPORT 5. </strong>

Train the SVM model by calling the trainSGDSVM method for 200 iterations (set learning rate parameter <em>η </em>to 0.01). Use the evaluateLinearModel to calculate and report the accuracy for the training and test data.

<strong>Problem 6: Comparing Linear Classifiers </strong>

Using the results for Problems 3–4, provide a comparison of the three different linear models (Perceptrons, Logistic Regression, and Support Vector Machines) on the provided data set.