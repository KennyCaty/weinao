from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired

class RegistForm(FlaskForm):
    username = StringField(label="账号", validators=[DataRequired()])
    fullname = StringField(label="姓名", validators=[DataRequired()])
    password = PasswordField(label="密码", validators=[DataRequired()])
    submit = SubmitField(label="注册")