from flask import Flask,Blueprint , render_template , url_for , redirect , request , session
import bcrypt
import random
import smtplib
import pymongo
import config
# from flask_pymongo import PyMongo


def create_app():
    app=Flask(__name__)
    # app.config.from_object('config')
    # app.config['MONGO_URI']="mongodb+srv://Karanam_Akhila:An12$shi@cluster0.h1vwyen.mongodb.net/?retryWrites=true&w=majority"
    # mongo=PyMongo(app)
    app.config['SECRET_KEY'] = config.SECRET_KEY

    # Set up email credentials
  
    from views import views
    app.register_blueprint(views,url_prefix="/")

    from auth import auth
    app.register_blueprint(auth,url_prefix="/")


    return app

if __name__=="__main__":
    app=create_app()
    app.run(debug=False,host='0.0.0.0')

    
