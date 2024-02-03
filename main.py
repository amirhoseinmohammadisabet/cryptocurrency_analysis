import collector as col
import models

def run():
    price = models.predict_currency_price("decision_tree", "btc", "tron", 33782.13, 107428757162,661077189473.97)
    print(price["predicted_price"])
    # 2018-07-30,6253.452283374108,107428757162.73953,1982310960.154297,0.02995057902787848,1969194013.7573497,210981453.09965715


if __name__ == "__main__":
    run()


