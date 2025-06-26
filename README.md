# ANN Salary Regression

This project demonstrates how to use an Artificial Neural Network (ANN) for predicting `EstimatedSalary` from the `Churn_Modelling.csv` dataset. The workflow includes data preprocessing, model building, training, evaluation, and saving both the model and preprocessing objects.

## Project Structure

- `salaryregression.ipynb` — Main notebook for salary regression using ANN.
- `Churn_Modelling.csv` — Dataset used for regression.
- `regression_model.h5` — Saved Keras regression model.
- `label_encoder_gender.pkl`, `onehot_encoder_geo.pkl`, `scaler.pkl` — Saved preprocessing objects.
- `regressionlogs/` — TensorBoard logs for training visualization.

## Workflow Overview

### 1. Data Preprocessing

- Drop irrelevant columns: `RowNumber`, `CustomerId`, `Surname`
- Encode categorical variables:
  - `Gender`: Label Encoding
  - `Geography`: One-Hot Encoding
- Scale features using `StandardScaler`
- Save encoders and scaler for future inference

### 2. Model Building

- Build a Sequential ANN model with Keras:
  - Input layer matches the number of features
  - Hidden layers: 64 and 32 neurons with ReLU activation
  - Output layer: 1 neuron (regression)
- Compile with Adam optimizer and `mean_absolute_error` loss

### 3. Training

- Use EarlyStopping to prevent overfitting
- Use TensorBoard for training visualization

### 4. Evaluation

- Evaluate the model on the test set:
  ```
  test_loss, test_mae = model.evaluate(X_test, y_test)
  print(f'Test loss: {test_loss}')
  print(f'Test MAE : {test_mae}')
  ```

### 5. Saving

- Save the trained model as `regression_model.h5`
- Save encoders and scaler as `.pkl` files

### 6. Visualization

- Launch TensorBoard to visualize training logs:
  ```
  tensorboard --logdir regressionlogs/fit
  ```

## Requirements

- Python 3.x(but x should be less than 11 because tensorflow does not work with python 3.12+)
- TensorFlow
- scikit-learn
- pandas
- numpy

Install dependencies with:

```
pip install -r requirements.txt
```

---

## Creator

- **Rudra Chouhan**
- **+91 7549019916**

## Credits

- This project based on tutorials by **Krish Naik Sir**.
