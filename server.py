from flask import Flask
from flask import render_template

app = Flask(__name__)

@app.route("/")
def welcome():
	return render_template('hello.html')

# @app.route("/hello")
# def hello():
# 	return render_template('hello.html')

@app.route("/start_session", methods=['GET', 'POST'])
def start_session():
    return "Start session"

@app.route("/end_session", methods=['GET', 'POST'])
def end_session():
	return "End session"

if __name__ == "__main__":
	app.debug = True
	app.run()