# holmusk

We have been given 4 datasets, each in CSV format.

clinical_data.csv contains medical information, such as symptoms and past medical records of the patient. Its primary key is id.
demographics.csv contains personal information of the patient, like age, gender, race etc. Its primary key is patient_id.
bill_id.csv contains a unique identifier for each medical bill, and the respective date of admission of the patient. Its primary key is bill_id. Its foreign key is patient_id.
bill_amount.csv contains the corresponding amount in dollars for each bill_id, which also happens to be its primary key.

Our end objective is as follows:
To build a solitary comprehensive dataframe using python, containing actionable information from all the above datasets. We need to figure out a way to join all the tables together in a meaningful, neat, and organised manner. This will facilitate smoother analysis of the data.
To conduct exploratory data analysis on the newly generated comprehensive dataframe. The end goal is to obtain insights behind the drivers responsible for medical costs.
