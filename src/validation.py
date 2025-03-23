from pandera import Check, Column, DataFrameSchema

medical_schema = DataFrameSchema({
    "patient_id": Column(str, Check.str_matches(r"^P\d{3}$")),
    "age": Column(int, Check.in_range(18, 120)),
    "bmi": Column(float, Check.in_range(15.0, 50.0)),
    "disease_risk": Column(int, Check.isin([0, 1]))
})

def validate_input(data):
    try:
        medical_schema.validate(data)
        return True, ""
    except Exception as e:
        return False, str(e)
