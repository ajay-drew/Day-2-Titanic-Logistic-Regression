# Titanic Logistic Regression Model

An intermediate-level logistic regression implementation using scikit-learn to predict passenger survival from the Titanic dataset, accessible on Kaggle. This project includes data preprocessing, feature scaling, model evaluation, and visualizations, all containerized with Docker for reproducibility.

### Features
- **Preprocessing**: Handles missing values and converts categorical variables
- **Feature Scaling**: Uses StandardScaler for normalization
- **Model**: LogisticRegression with L2 regularization and custom parameters
- **Evaluation**: Provides accuracy score and detailed classification report
- **Visualizations**: Feature importance bar plot saved as PNG
- **Docker Setup**: Containerized environment for consistent execution

## Docker Setup

### Build the Docker Image

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ajay-drew/titanic-survival-regression.git
   cd titanic-survival-regression
   ```

2. **Build the Docker Image**
    ```bash
    docker build -t titanic-logistic-regression .
    ```

3. **Run the Container**
    ```bash
    docker run -v /full/path/to/titanic-survival-regression/output:/app/output titanic-logistic-regression
    ```