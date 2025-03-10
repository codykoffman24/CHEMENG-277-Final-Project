import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt 

# Generate example data
np.random.seed(42)
n_samples = 100

# Create synthetic predictor variables
X = np.random.randn(n_samples , 3) # Temperature , Pressure , Catalyst
# True coefficients
beta_true = np . array ([2.5 , -1.5 , 3.0])
# Generate target with some noise
y = np.dot (X , beta_true ) + 0.5 * np . random . randn ( n_samples )

def fit_linear_regression(X , y ):
    #Add column of ones for intercept
    X_b = np.column_stack ([ np . ones ( len ( X ) ) , X ])

    # Calculate beta using normal equation
    # \ beta = ( X ^ T X ) ^( -1) X ^ T y
    beta = np.linalg.inv ( X_b . T . dot ( X_b ) ) . dot ( X_b . T ) . dot ( y )

    return beta

 # Fit the model
beta_manual = fit_linear_regression(X , y )

print("Manual Implementation Results : " )
print(f"Intercept : {beta_manual [0]:.3f}")
print(f"Coefficients : { beta_manual [1:].round(3)}")



# Split data into training and testing set
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size =0.2 , random_state =42)

# Create and fit the model
model = LinearRegression()
model.fit( X_train , y_train )

# Make predictions
y_pred = model.predict( X_test )

print( " \ nscikit - learn Implementation Results : ")
print(f" Intercept : {model.intercept_ :.3} ")
print(f" Coefficients : {model.coef_ . round (3) } ")
print(f" R Score : {r2_score(y_test , y_pred):.3} ")


def plot_diagnostics(y_true , y_pred , residuals):
    fig,(ax1 , ax2) = plt.subplots(1 , 2 , figsize =(12 , 4))
    # Predicted vs Actual
    ax1.scatter(y_pred , y_true)
    ax1.plot([ y_true.min() , y_true.max()] , [ y_true.min() , y_true.max() ] , 'r--', lw =2)
    ax1.set_xlabel('Predicted Values')
    ax1.set_ylabel('Actual Values')
    ax1.set_title('Predicted vs Actual Values')

    # Residual Plot
    ax2.scatter( y_pred , residuals )
    ax2.axhline(y =0 , color = 'r' , linestyle = '--')
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residual Plot')
                    
    plt.tight_layout()

    return fig

# Calculate residuals
residuals = y_test - y_pred

# Create diagnostic plots
plot_diagnostics(y_test, y_pred, residuals)
plt.show()