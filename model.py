from wtforms import Form, FloatField, validators

# 'pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label'
class InputForm(Form):
    a = FloatField(
        label='pregnant', default=1,
        validators=[validators.InputRequired()])
    b = FloatField(
        label='glucose', default=1,
        validators=[validators.InputRequired()])
    c = FloatField(
        label='bp', default=1,
        validators=[validators.InputRequired()])
    d = FloatField(
        label='skin', default=1,
        validators=[validators.InputRequired()])

    e = FloatField(
        label='insulin', default=1,
        validators=[validators.InputRequired()])
    z = FloatField(
        label='bmi', default=1,
        validators=[validators.InputRequired()])
    g = FloatField(
        label='pedigree ', default=1,
        validators=[validators.InputRequired()])
    h = FloatField(
        label='age ', default=1,
        validators=[validators.InputRequired()])
