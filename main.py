import collector as col
import models

def run():
    price = models.predict_currency_price("decision_tree", "btc", "tron", 33782.13, 107428757162,661077189473.97)
    print(price["predicted_price"])

    # col.crypto_for_clustering()
    models.choosing_four()

if __name__ == "__main__":
    run()


