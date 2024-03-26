from sqlalchemy import select
from models.user import User
from routes import db
from flask_login import login_user # 这个方法可以通过LoginManager完成往session里面放用户信息的操作

class UserService:
    def do_login(self, username, password):
        query = select(User).where(User.username == username)
        attempt_user = db.session.scalar(query)
        if attempt_user and attempt_user.check_password_corr(attempt_password=password) :
            login_user(attempt_user) ####
            return True
        
        return False
    
    def do_regist(self, username, fullname, password):
        query = select(User).where(User.username == username)  # 查找有没有注册过
        finded_user = db.session.scalar(query)
        if finded_user:
            return False
        else:
            u = User(username=username, fullname=fullname, password=password)
            db.session.add(u)
            db.session.commit()
            return True
            