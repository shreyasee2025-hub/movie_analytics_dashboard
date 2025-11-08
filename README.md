Project: Azure Movie Analytics Dashboard

Student: Shreyasee Poddar

Course: DSA 508 - Big Data Platforms

Professor: Daniel Hernando Romero Rodriguez Sr.

Contents:

1. dashboard.py  - Streamlit dashboard connected to Azure Cosmos DB (Mongo API)
   
2. Azure_Movie_Analytics_Narrative.pdf - Explanatory narrative
   
3. requirements.txt - Python dependencies
   
4. .streamlit/config.toml - Dashboard theme configuration

Database Info:

Cosmos DB cluster: cosmos-shrey-mflix.global.mongocluster.cosmos.azure.com

Database connection (evidence):

• The app connects to Azure Cosmos DB (Mongo API) and shows “Connected successfully to Azure Cosmos DB!” in the Overview tab.

• Database name: sample_mflix (same dataset as Problem 1, as per instructions allowing existing database reuse).

To run locally:

> pip install -r requirements.txt

> streamlit run dashboard.py

Notes:

(i) Secrets (MONGO_URI, DB_NAME) are stored in Streamlit Cloud -> App Settings -> Secrets

(ii) I used: DB_NAME = "sample_mflix" (as in Problem 1)

