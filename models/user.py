from datetime import datetime

from routes import db, login_manager
from sqlalchemy import Integer, String, BLOB, TIMESTAMP, Column
from sqlalchemy.orm import  Mapped
from flask_login import UserMixin

# flask_login需要指定是什么查到用户的
@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, user_id)



# UserMixin 登陆验证会用到一些特殊的方法，在不修改的情况下可以直接用已经实现的轮子
class User(db.Model, UserMixin):
    """
        用户
    """
    __tablename__ = 'users'
    id: Mapped[int] = Column(Integer, primary_key=True)
    username: Mapped[str] = Column(String, nullable=False, unique=True)
    fullname: Mapped[str] = Column(String, nullable=False)
    password: Mapped[str] = Column(String, nullable=False)
    description: Mapped[str] = Column(String, nullable=True)
    
    def check_password_corr(self, attempt_password):
        return self.password == attempt_password