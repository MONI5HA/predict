# Import required libraries
import os
from flask import Flask, render_template  # Flask for web framework
import pandas as pd  # Pandas for data handling
import matplotlib.pyplot as plt  # Matplotlib for plotting
import seaborn as sns  # Seaborn for advanced visualizations
import numpy as np  # Numpy for numerical computations
from sklearn.model_selection import train_test_split  # Splitting data for model training
from sklearn.linear_model import LinearRegression  # Linear regression model
from sklearn.ensemble import RandomForestRegressor  # Random forest regression model
from sklearn.metrics import r2_score  # R-squared score for evaluation
import plotly.graph_objects as go  # Plotly for interactive charts

# Flask app initialization
app = Flask(__name__)

# Define the folder where static images will be saved
static_folder = os.path.join(app.root_path, 'static/images')

# Create the static/images directory if it does not exist
os.makedirs(static_folder, exist_ok=True)

# Load Excel and CSV data files
excel_data_path = 'data.xlsx'  # Path to the Excel file
csv_data_path = 'data2.csv'  # Path to the CSV file

# Load Excel file and parse the sheet
excel_data = pd.ExcelFile(excel_data_path)
nutrition_data = excel_data.parse('TDS Composite Descriptions')

# Load CSV file for disease data
disease_data = pd.read_csv(csv_data_path)

# Filter data for Canada
nutrition_data = nutrition_data[nutrition_data['Country'] == 'Canada']
disease_data = disease_data[disease_data['Country'] == 'Canada']

# Clean the 'Number of deaths' column by removing commas and converting to float
disease_data['Number of deaths'] = disease_data['Number of deaths'].str.replace(',', '').astype(float)

# Define population data for Canada
total_population = "41,288,599"
total_men = "20,638,255"
total_women = "20,650,344"

# Define relevant nutritional columns for analysis
nutritional_columns = [
    "Calories", "Carbohydrates", "Sugars", "Fiber", "Proteins", "Fats",
    "Saturated Fats", "Trans Fats", "Unsaturated Fats", "Vitamin A",
    "Vitamin C", "Vitamin D", "Vitamin E", "Vitamin K", "B Vitamins",
    "Calcium", "Iron", "Magnesium", "Potassium", "Sodium", "Zinc",
    "Cholesterol (mg)", "Omega-3"
]

# Convert the nutritional columns to numeric, ignoring errors
nutrition_data[nutritional_columns] = nutrition_data[nutritional_columns].apply(pd.to_numeric, errors='coerce')

# Function to save a plot to the static folder
def save_plot(plot_func, filename, **kwargs):
    plot_func(**kwargs)
    plt.tight_layout()
    plt.savefig(os.path.join(static_folder, filename))
    plt.close()

# Function to generate prediction charts
def generate_prediction_charts():
    # Calculate total nutritional content for each food item
    nutrition_data['Total Nutritional Content'] = nutrition_data[nutritional_columns].sum(axis=1)
    
    # Generate chart for top 10 foods based on total nutritional content
    top_foods = nutrition_data[['Composite Description (TDS_FC_Label)', 'Total Nutritional Content']] \
        .sort_values(by='Total Nutritional Content', ascending=False).head(10)
    save_plot(lambda: sns.barplot(x='Total Nutritional Content', y='Composite Description (TDS_FC_Label)', 
                                  data=top_foods, palette='viridis'), 'top_foods.png')

    # Generate chart for top nutrients
    total_nutrients = nutrition_data[nutritional_columns].sum().sort_values(ascending=False)
    save_plot(lambda: sns.barplot(x=total_nutrients.values, y=total_nutrients.index, palette="mako"),
              'top_nutrients.png')

    # Generate chart for average nutrients
    average_nutrients = nutrition_data[nutritional_columns].mean().sort_values(ascending=False)
    save_plot(lambda: sns.barplot(x=average_nutrients.values, y=average_nutrients.index, palette="coolwarm"),
              'average_nutrients.png')

    # Generate chart for nutritional deficiencies
    deficiency_to_count = disease_data['Nutrient  Deficiency'].str.split(',').explode().str.strip().value_counts()
    save_plot(lambda: sns.barplot(x=deficiency_to_count.values, y=deficiency_to_count.index, palette="flare"),
              'nutritional_deficiencies.png')

    # Generate chart for nutritional contributors
    nutritional_contributors = (total_nutrients / total_nutrients.sum() * 100).sort_values(ascending=False)
    save_plot(lambda: sns.barplot(x=nutritional_contributors.values, y=nutritional_contributors.index, palette="cubehelix"),
              'nutritional_contributors.png')

# Function to train models and predict top diseases based on deficiencies
def get_top_diseases():
    # Drop rows with missing nutrient deficiency data
    disease_data_cleaned = disease_data.dropna(subset=['Nutrient  Deficiency'])

    # Aggregate nutrient data to summarize deficiencies
    nutrient_aggregates = nutrition_data.mean(numeric_only=True)

    # Match deficiencies with nutrient averages
    disease_data_cleaned['Deficiencies'] = disease_data_cleaned['Nutrient  Deficiency'].apply(
        lambda x: [nutrient_aggregates.get(n.strip(), 0) for n in (x.split(',') if pd.notna(x) else [])]
    )
    disease_data_cleaned['Deficiencies'] = disease_data_cleaned['Deficiencies'].apply(
        lambda x: np.mean(x) if x else 0
    )

    # Split data into training and testing sets
    X = np.array(disease_data_cleaned['Deficiencies']).reshape(-1, 1)
    y = disease_data_cleaned['Number of deaths']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Linear Regression and Random Forest models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42)
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        r2 = r2_score(y_test, predictions)
        results[name] = r2

    # Select the best model and fit on the entire dataset
    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]
    best_model.fit(X, y)

    # Predict likelihood of diseases
    disease_data_cleaned['Predicted Likelihood'] = best_model.predict(X)

    # Return top 10 diseases with the highest likelihood
    return disease_data_cleaned.sort_values(by='Predicted Likelihood', ascending=False).head(10)

# Function to calculate total nutrient consumption
def calculate_totals():
    totals = nutrition_data[nutritional_columns].sum()
    return totals.to_dict()

# Function to create a speedometer chart using Plotly
def create_speedometer(value, title, max_value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        gauge={
            "axis": {"range": [0, max_value]},
            "bar": {"color": "blue"},
            "steps": [
                {"range": [0, max_value * 0.5], "color": "lightgreen"},
                {"range": [max_value * 0.5, max_value * 0.75], "color": "yellow"},
                {"range": [max_value * 0.75, max_value], "color": "red"}
            ]
        },
        title={"text": title}
    ))
    return fig.to_html(full_html=False)

# Flask route for the index page
@app.route('/')
def index():
    generate_prediction_charts()  # Generate all prediction charts
    top_diseases = get_top_diseases()  # Fetch top diseases
    return render_template('index.html', top_diseases=top_diseases, 
                           total_population=total_population, 
                           total_men=total_men, 
                           total_women=total_women)
# Function to find nutrient-rich foods
def find_nutrient_rich_foods(nutrient):
    nutrient = nutrient.strip().lower()  # Normalize nutrient name
    matching_columns = [col for col in nutrition_data.columns if nutrient in col.lower()]
    
    if matching_columns:
        column = matching_columns[0]  # Take the first match
        top_foods = nutrition_data[['Composite Description (TDS_FC_Label)', column]] \
                    .sort_values(by=column, ascending=False).head(5)
        return top_foods.rename(columns={column: f'{nutrient.capitalize()} Content'})
    else:
        # Return an empty DataFrame if no matching column is found
        return pd.DataFrame(columns=['Composite Description (TDS_FC_Label)', f'{nutrient.capitalize()} Content'])

# Function to generate recommendations
def generate_recommendations():
    disease_data_cleaned = disease_data.dropna(subset=['Nutrient  Deficiency'])
    recommendations = {}
    for _, row in disease_data_cleaned.iterrows():
        deficiencies = row['Nutrient  Deficiency'].split(',')
        foods = {deficiency: find_nutrient_rich_foods(deficiency).to_dict(orient='records') for deficiency in deficiencies}
        recommendations[row['Disease Name']] = foods
    return recommendations

# Flask route for recommendations
@app.route('/recommendations')
def recommendations():
     # Calculate totals
    totals = calculate_totals()
    calories_chart = create_speedometer(totals['Calories'], "Total Calories Consumed", 2500)
    proteins_chart = create_speedometer(totals['Proteins'], "Total Proteins Consumed (g)", 150)
    carbs_chart = create_speedometer(totals['Carbohydrates'], "Total Carbohydrates Consumed (g)", 300)
    sugars_chart = create_speedometer(totals['Sugars'], "Total Sugars Consumed (g)", 50)

    recommendations = generate_recommendations()  # Generate recommendations dynamically
    return render_template('recommendations.html', recommendations=recommendations,calories_chart=calories_chart, 
                           proteins_chart=proteins_chart, 
                           carbs_chart=carbs_chart, 
                           sugars_chart=sugars_chart)

if __name__ == '__main__':
    app.run(debug=True)
