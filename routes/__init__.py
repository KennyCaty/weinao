from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager

app = Flask(__name__, template_folder='../templates',
            static_folder='../static',
            static_url_path='/static')

#MSOL所在的主机名
HOSTNAME="127.0.0.1"
#MySQL监听的端口号，默认3306
PORT=3306
#连接MySQL的用户名，用自己设置的
USERNAME="root"
#连剂ySQL的密码，读者用自己的
PASSWORD="123456"
#MySQL上创建的数据库名称
DATABASE="ict"

# mysql+mysqldb://<user>:<password>@<host>[:<port>]/<dbname>
app.config['SQLALCHEMY_DATABASE_URI'] = f"mysql+mysqldb://{USERNAME}:{PASSWORD}@{HOSTNAME}:{PORT}/{DATABASE}?charset=utf8"
app.config['SECRET_KEY'] = '123456' # 随机生成就行

app.config['UPLOAD_FOLDER'] = './upload/'

db = SQLAlchemy(app)
# 用于处理登录保存session的
login_manager = LoginManager(app)





from routes import user_routes
from routes import predict_routes
# from routes import admin_routes

