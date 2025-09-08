import pandas as pd
import logging
import pycountry

# Configure logging for debugging (currently commented out for cleaner execution)
# logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

# Load the dataset from a CSV file
file_path = "covid_19_set_2.csv"
df = pd.read_csv(file_path)

# Convert specific columns to numeric types, coercing invalid values to NaN
numeric_cols = ["Lat", "Long", "Confirmed", "Deaths", "Recovered", "Active"]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Rows where the latitude and longitude are not in the valid ranges
invalid_lat_long = df[(df["Lat"].abs() > 90) | (df["Long"].abs() > 180)][["ID", "Lat", "Long"]]

# Rows where Confirmed, Deaths, Recovered, or Active cases have negative values
negative_values = df[(df[["Confirmed", "Deaths", "Recovered", "Active"]] < 0).any(axis=1)]

# Checking the formula: Active = Confirmed - Deaths - Recovered stores accurate values
df["Computed_Active"] = df["Confirmed"] - df["Deaths"] - df["Recovered"]

# Rows where the reported Active cases do not match the computed Active cases
active_mismatch = df[df["Active"] != df["Computed_Active"]]

# All empty rows (excluding the ID column)
missing_rows = df[df.isnull().all(axis=1)]

# Using the pycountry package to get the UN list of valid country names to compare the country/region column to
valid_countries = {c.name.lower(): c.name for c in pycountry.countries}

# Certain valid countries/regions are listed differently in the UN naming in the pycountry package
# Manually add these limited countries to be classified as being valid and not polluted
country_aliases = {
    "bolivia": "bolivia, plurinational state of",
    "south korea": "korea, republic of",
    "north korea": "korea, democratic people's republic of",
    "russia": "russian federation",
    "venezuela": "venezuela, bolivarian republic of",
    "vietnam": "viet nam",
    "syria": "syrian arab republic",
    "tanzania": "tanzania, united republic of",
    "palestine": "palestine, state of",
    "laos": "lao people's democratic republic",
    "moldova": "moldova, republic of",
    "taiwan": "taiwan, province of china",
    "burma": "myanmar",
    "cape verde": "cabo verde",
    "czech republic": "czechia",
    "ivory coast": "côte d'ivoire",
    "macedonia": "north macedonia",
    "east timor": "timor-leste",
    "swaziland": "eswatini",
    "iran": "iran, islamic republic of",
    "yemen": "yemen, republic of",
    "libya": "libya, state of",
    "kosovo": "Kosovo",
    "brunei": "Brunei Darussalam",
    "congo (brazzaville)": "Congo",
    "congo (kinshasa)": "Congo, the Democratic Republic of the",
    "cote d'ivoire": "Côte d'Ivoire",
    "west bank and gaza": "Palestine, State of",
    "us": "United States",
    "El  Salvador": "El Salvador"
}


# Function to normalize country names using the valid country list and aliases
def normalize_country_name(country_name):
    country_name = country_name.strip().lower()
    if country_name in country_aliases:
        return country_aliases[country_name]  # Return standardized alias name if available
    return valid_countries.get(country_name, None)  # Otherwise, return the official name if valid

# Identify invalid or missing country names
polluted_countries = []
for index, row in df.iterrows():
    country = row['Country/Region']

    if pd.isna(country) or country.strip() == "":  # Check for missing or empty country name
        polluted_countries.append([row["ID"], "Country/Region", "Missing or empty country"])
    else:
        normalized_name = normalize_country_name(country)
        if normalized_name is None:  # If country is unrecognized
            logging.debug(f"Invalid country detected: '{country}' (Row ID: {row['ID']})")
            polluted_countries.append([row["ID"], "Country/Region", "Invalid or inconsistent country name"])

# Define valid WHO Regions for verification
who_regions = {"Africa", "Americas", "Eastern Mediterranean", "Europe", "South-East Asia", "Western Pacific"}

# Identify rows with incorrect WHO Region names
invalid_who_region = df[~df["WHO Region"].isin(who_regions) & df["WHO Region"].notna()]

# Collect all polluted data entries
pollution_records = []

# Add polluted invalid latitude/longitude entries
for _, row in invalid_lat_long.iterrows():
    pollution_records.append([row["ID"], "Lat", "Latitude out of bounds"])
    pollution_records.append([row["ID"], "Long", "Longitude out of bounds"])

# Add polluted negative values in numeric fields
for _, row in negative_values.iterrows():
    if row["Confirmed"] < 0:
        pollution_records.append([row["ID"], "Confirmed", "Negative value"])
    if row["Deaths"] < 0:
        pollution_records.append([row["ID"], "Deaths", "Negative value"])
    if row["Recovered"] < 0:
        pollution_records.append([row["ID"], "Recovered", "Negative value"])
    if row["Active"] < 0:
        pollution_records.append([row["ID"], "Active", "Negative value"])

# Add polluted cases where Active cases do not match the computed value
for _, row in active_mismatch.iterrows():
    pollution_records.append([row["ID"], "Active", "Active case equation mismatch"])

# add polluted completely missing rows
for _, row in missing_rows.iterrows():
    pollution_records.append([row["ID"], "all columns", "Entire row missing"])

# add polluted detected country inconsistencies
for entry in polluted_countries:
    pollution_records.append(entry)

# add polluted invalid WHO Region entries
for _, row in invalid_who_region.iterrows():
    pollution_records.append([row["ID"], "WHO Region", "Misspelled or incorrect WHO region"])

# Convert collected pollution records into a DataFrame
pollution_df = pd.DataFrame(pollution_records, columns=["ID", "Column", "Description"])

# Upload the polluted entries to a solution CSV based on assignment guidelines
solution_file_path = "solution_pollution.csv"
pollution_df.to_csv(solution_file_path, index=False)

# End of program message
print(f"Pollution report saved as {solution_file_path}")