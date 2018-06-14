import pickle
from scipy.sparse import hstack


class_names = ["toxic",
               "severe_toxic",
               "obscene",
               "threat",
               "insult",
               "identity_hate"]


class Judge_toxic():
    def __init__(self):
        self.classifiers = {}
        for class_name in class_names:
            with open("../pickles/{}.pickle".format(class_name), "rb") as f:
                self.classifiers[class_name] = pickle.load(f)

        with open("../pickles/word_vectorizer.pickle", "rb") as f:
            self.word_vectorizer = pickle.load(f)

        with open("../pickles/char_vectorizer.pickle", "rb") as f:
            self.char_vectorizer = pickle.load(f)

    def read_input(self):
        self.text = [input()]

    def predict_toxic(self):
        word_features = self.word_vectorizer.transform(self.text)
        char_features = self.char_vectorizer.transform(self.text)
        features = hstack([char_features, word_features])
        results = {}
        for class_name in class_names:
            proba = self.classifiers[class_name].predict_proba(features)[0][1]
            results[class_name] = proba
        return results


def main():
    judge_toxic = Judge_toxic()
    print("You can enter comments 3 times")
    for _ in range(3):
        judge_toxic.read_input()
        results = judge_toxic.predict_toxic()
        print(results)


if __name__ == "__main__":
    main()
