
from routes import app
from flask import render_template, abort, redirect, flash, url_for
# from services.article_service import ArticleService
from services.user_service import UserService
from forms.login_form import LoginForm
from forms.resgist_form import RegistForm
from flask_login import logout_user



@app.route('/')
@app.route('/index')
def home_page():
    # artcles = ArticleService().get_articles()

    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login_page():
    form = LoginForm()
    if form.validate_on_submit(): # 如果是True，则说明是post操作
        result = UserService().do_login(username=form.username.data, password=form.password.data)
        if result:
            flash(f'欢迎回来', category='success')
            return redirect(url_for('home_page'))
        else:
            flash('用户名或密码错误，请重新输入！', category='danger')
    return render_template('login.html', form=form)



@app.route('/regist', methods=['GET', 'POST'])
def regist_page():
    form = RegistForm()
    if form.validate_on_submit(): # 如果是True，则说明是post操作
        result = UserService().do_regist(username=form.username.data, fullname=form.fullname.data, password=form.password.data)
        if result:
            flash(f'注册成功，请登录', category='success')
            return redirect(url_for('login_page'))
        else:
            flash('用户名已注册，注册失败', category='danger')
    return render_template('regist.html', form=form)



@app.route('/logout')
def logout():
    logout_user() # flask_login提供，不需要重复造轮子
    return redirect(url_for('home_page'))

@app.route('/about')
def about_page():
    return render_template('about.html')

# @app.routes('/article')