from flask import Blueprint,render_template
views=Blueprint("views",__name__)

@views.route("/")
@views.route("/main")
def main():
    return render_template("Main.html")