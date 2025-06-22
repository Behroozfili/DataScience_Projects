
import numpy as np 
import pandas as pd

# Definition von Datentypen
my_string = "Hello, world!"
my_float = 3.14
my_integer = 42
my_boolean = True
my_list = [1, 2, 3, 4]
my_tuple = (10, 20, 30)
my_dict = {"name": "Stefan", "age": 25}
my_set = {1, 2, 3, 3, 2} 
my_array = np.array([1, 2, 3, 4, 5])

# Erstellen eines DataFrames mit Pandas
def create_dataframe():
    data = {
        "Name": ["Stefan ", "Sara"],
        "Age": [28, 34]
    }
    return pd.DataFrame(data)

# Ausgabe der Daten
def print_data():
    print("String:", my_string)
    print("Float:", my_float)
    print("Integer:", my_integer)
    print("Boolean:", my_boolean)
    print("List:", my_list)
    print("Tuple:", my_tuple)
    print("Dictionary:", my_dict)
    print("Set:", my_set)
    print("Numpy Array:", my_array)
    print("Pandas DataFrame:\n", create_dataframe())

# Erstellen von Patientendaten
def create_patient_data():
    np.random.seed(42)
    return pd.DataFrame({
        'patient_id': range(1, 1001),
        'doctor_id': np.random.randint(1, 51, size=1000),
        'treatment_id': np.random.randint(1, 21, size=1000),
        'dosage_mg': np.random.randint(50, 501, size=1000),
        'visit_date': pd.date_range(start='2024-01-01', periods=91).repeat(11)[:1000]
    })

# Erstellen von Behandlungsdaten
def create_treatment_data():
    return pd.DataFrame({
        'treatment_id': range(1, 21),
        'treatment_name': [f'Drug_{letter}' for letter in 'ABCDEFGHIJKLMNOPQRST'],
        'efficacy_score': np.round(np.random.uniform(0.5, 0.95, size=20), 2),
        'category': np.random.choice(['Antibiotics', 'Anti-inflammatory', 'Hormonal', 'Antiviral'], size=20)
    })

# a) Berechnung der Gesamtdosierung für jede Behandlung
def calculate_total_dosage(patients_df):
    return patients_df.groupby('treatment_id')['dosage_mg'].sum().reset_index()

# b) Finden der Top 5 Behandlungen nach insgesamt verschriebener Dosierung
def top_5_treatments_by_dosage(total_dosage_per_treatment):
    return total_dosage_per_treatment.sort_values(by='dosage_mg', ascending=False).head(5)

# a) Zusammenführen der Patienten- und Behandlungsdaten
def merge_patient_and_treatment_data(patients_df, treatments_df):
    return pd.merge(patients_df, treatments_df, on='treatment_id', how='left')

# b) Berechnung der durchschnittlichen Dosierung nach Behandlungsgruppe
def calculate_avg_dosage_by_category(merged_df):
    return merged_df.groupby('category')['dosage_mg'].mean().reset_index()

# c) Finden der Behandlungsgruppe mit der höchsten durchschnittlichen Wirksamkeit
def best_category_by_efficacy(treatments_df):
    avg_efficacy_by_category = treatments_df.groupby('category')['efficacy_score'].mean().reset_index()
    return avg_efficacy_by_category.sort_values(by='efficacy_score', ascending=False).head(1)

# a) Berechnung der täglichen Patientenbesuche
def calculate_daily_visits(patients_df):
    return patients_df.groupby('visit_date').size().reset_index(name='visit_count')

# b) Finden des Tages mit den meisten Patientenbesuchen
def busiest_day(daily_visits):
    return daily_visits.sort_values(by='visit_count', ascending=False).head(1)

# c) Erstellen einer wöchentlichen Zusammenfassung der Behandlungen nach Kategorie
def weekly_summary_by_category(merged_df):
    merged_df['week'] = merged_df['visit_date'].dt.to_period('W').apply(lambda r: r.start_time)
    return merged_df.groupby(['week', 'category']).size().reset_index(name='treatment_count')

# a) Berechnung der kumulierten Dosierung für jede Behandlung
def calculate_running_total_dosage(merged_df):
    merged_df_sorted = merged_df.sort_values(['treatment_id', 'visit_date'])
    merged_df_sorted['running_total_dosage'] = merged_df_sorted.groupby('treatment_id')['dosage_mg'].cumsum()
    return merged_df_sorted[['treatment_id', 'visit_date', 'dosage_mg', 'running_total_dosage']]

# b) Finden von Ärzten, die mindestens 3 verschiedene Behandlungsarten verschrieben haben
def find_active_doctors(merged_df):
    doctor_categories = merged_df.groupby('doctor_id')['category'].nunique().reset_index()
    return doctor_categories[doctor_categories['category'] >= 3]

# c) Identifizierung von Behandlungen, die jede Woche im Datensatz verschrieben wurden
def treatments_prescribed_every_week(merged_df):
    all_weeks = merged_df['week'].nunique()
    weeks_per_treatment = merged_df.groupby('treatment_id')['week'].nunique().reset_index()
    return weeks_per_treatment[weeks_per_treatment['week'] == all_weeks]

print_data()
patients_df = create_patient_data()
treatments_df = create_treatment_data()
total_dosage_per_treatment = calculate_total_dosage(patients_df)
print("\nTotal dosage per treatment_id:")
print(total_dosage_per_treatment.head())

top_5_treatments = top_5_treatments_by_dosage(total_dosage_per_treatment)
print("\nTop 5 treatments by total dosage prescribed:")
print(top_5_treatments)

merged_df = merge_patient_and_treatment_data(patients_df, treatments_df)
avg_dosage_by_category = calculate_avg_dosage_by_category(merged_df)
print("\nAverage dosage by treatment category:")
print(avg_dosage_by_category)

best_category = best_category_by_efficacy(treatments_df)
print("\nCategory with the highest average efficacy score:")
print(best_category)

daily_visits = calculate_daily_visits(patients_df)
print("\nDaily patient visit totals:")
print(daily_visits.head())

busiest_day_result = busiest_day(daily_visits)
print("\nDay with the highest number of patient visits:")
print(busiest_day_result)

weekly_summary = weekly_summary_by_category(merged_df)
print("\nWeekly summary of treatments by category:")
print(weekly_summary.head())

running_total_dosage = calculate_running_total_dosage(merged_df)
print("\nRunning total of dosage prescribed for each treatment:")
print(running_total_dosage.head())

active_doctors = find_active_doctors(merged_df)
print("\nDoctors who have prescribed at least 3 different treatment categories:")
print(active_doctors)

always_prescribed = treatments_prescribed_every_week(merged_df)
print("\nTreatments prescribed every week throughout the dataset:")
print(always_prescribed)
