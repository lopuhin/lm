import argparse
from typing import List

from flask import Flask, render_template, request
from nltk.tokenize import wordpunct_tokenize

from predict import Model


app = Flask(__name__)
model = None  # type: Model


@app.route('/')
def main():
    phrase = request.args.get('phrase')
    top = []
    if phrase:
        tokens = tokenize(phrase)
        top = model.predict_top(tokens)
    return render_template(
        'main.html',
        phrase=phrase,
        top=top,
        )


def tokenize(phrase: str) -> List[str]:
    return [token.lower() for token in wordpunct_tokenize(phrase)]


def run():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('model')
    arg('vocab')
    arg('--port', type=int, default=8000)
    arg('--host', default='localhost')
    arg('--debug', action='store_true')
    args = parser.parse_args()

    global model
    model = Model(args.model, args.vocab)
    app.run(port=args.port, host=args.host, debug=args.debug, threaded=False)


if __name__ == '__main__':
    run()
