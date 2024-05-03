import collector as col
import models

def run():

    # col.crypto_for_clustering()
    models.choosing_four(1)
    models.top4_correlation()
    models.top4_eda()
    models.all()
    models.my_lstm()

if __name__ == "__main__":
    run()


