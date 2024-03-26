from models.article import Article
from routes import db
from sqlalchemy import Select

class ArticleService:
    def get_article(self, id):
        """
        根据Id查询文章
        :param id:
        :return:
        """
        return db.session.get(Article, id)

    def get_articles(self):
        query = Select(Article)
        return db.session.scalars(query).all()