import os
from flask import Flask, render_template
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

app = Flask(__name__)
static_folder = os.path.join(app.root_path, 'static/images')

# Ensure the static/images directory exists
os.makedirs(static_folder, exist_ok=True)

# Load Data
excel_data_path = 'data.xlsx'  # Replace with your actual path
csv_data_path = 'data2.csv'  # Replace with your actual path

excel_data = pd.ExcelFile(excel_data_path)
nutrition_data = excel_data.parse('TDS Composite Descriptions')
disease_data = pd.read_csv(csv_data_path)

# Filter for Canada
nutrition_data = nutrition_data[nutrition_data['Country'] == 'Canada']
disease_data = disease_data[disease_data['Country'] == 'Canada']

# Clean 'Number of deaths'
disease_data['Number of deaths'] = disease_data['Number of deaths'].str.replace(',', '').astype(float)

# Calculate total population
total_population = "41,288,599"
total_men = "20,638,255"
total_women = "20,650,344"

# Nutritional Columns
nutritional_columns = [
    "Calories", "Carbohydrates", "Sugars", "Fiber", "Proteins", "Fats",
    "Saturated Fats", "Trans Fats", "Unsaturated Fats", "Vitamin A",
    "Vitamin C", "Vitamin D", "Vitamin E", "Vitamin K", "B Vitamins",
    "Calcium", "Iron", "Magnesium", "Potassium", "Sodium", "Zinc",
    "Cholesterol (mg)", "Omega-3"
]
nutrition_data[nutritional_columns] = nutrition_data[nutritional_columns].apply(pd.to_numeric, errors='coerce')

# Save KPI Charts
def save_plot(plot_func, filename, **kwargs):
    plot_func(**kwargs)
    plt.tight_layout()
    plt.savefig(os.path.join(static_folder, filename))
    plt.close()

# Prediction Charts
def generate_prediction_charts():
    # Top Foods
    nutrition_data['Total Nutritional Content'] = nutrition_data[nutritional_columns].sum(axis=1)
    top_foods = nutrition_data[['Composite Description (TDS_FC_Label)', 'Total Nutritional Content']] \
        .sort_values(by='Total Nutritional Content', ascending=False).head(10)
    save_plot(lambda: sns.barplot(x='Total Nutritional Content', y='Composite Description (TDS_FC_Label)', 
                                  data=top_foods, palette='viridis'), 'top_foods.png')

    # Top Nutrients
    total_nutrients = nutrition_data[nutritional_columns].sum().sort_values(ascending=False)
    save_plot(lambda: sns.barplot(x=total_nutrients.values, y=total_nutrients.index, palette="mako"),
              'top_nutrients.png')

    # Average Nutrients
    average_nutrients = nutrition_data[nutritional_columns].mean().sort_values(ascending=False)
    save_plot(lambda: sns.barplot(x=average_nutrients.values, y=average_nutrients.index, palette="coolwarm"),
              'average_nutrients.png')

    # Nutritional Deficiencies
    deficiency_to_count = disease_data['Nutrient  Deficiency'].str.split(',').explode().str.strip().value_counts()
    save_plot(lambda: sns.barplot(x=deficiency_to_count.values, y=deficiency_to_count.index, palette="flare"),
              'nutritional_deficiencies.png')

    # Nutritional Contributors
    nutritional_contributors = (total_nutrients / total_nutrients.sum() * 100).sort_values(ascending=False)
    save_plot(lambda: sns.barplot(x=nutritional_contributors.values, y=nutritional_contributors.index, palette="cubehelix"),
              'nutritional_contributors.png')

def get_top_diseases():
    # Predictions for Nutritional Deficiencies
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

    # Prepare features and target variables
    X = np.array(disease_data_cleaned['Deficiencies']).reshape(-1, 1)
    y = disease_data_cleaned['Number of deaths']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train models and evaluate
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

    # Select the best model
    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]
    best_model.fit(X, y)

    # Predict the likelihood of diseases
    disease_data_cleaned['Predicted Likelihood'] = best_model.predict(X)

    # Get top diseases
    return disease_data_cleaned.sort_values(by='Predicted Likelihood', ascending=False).head(10)

# Flask Routes
@app.route('/')
def index():
    generate_prediction_charts()  # Generate prediction charts dynamically
    top_diseases = get_top_diseases()  # Fetch top diseases
    return render_template('index.html', top_diseases=top_diseases,total_population=total_population,total_men = total_men,total_women = total_women)
  
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
    recommendations = generate_recommendations()  # Generate recommendations dynamically
    return render_template('recommendations.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
