import collector as col
import models

def run():
    try:
        data = col.read_data_pd("Data/data.csv")
        # print(data)
        model1, scaler1= models.knn_model()
        # model2, scaler2 = ann_model()
        input_data = [0.1, 7194154492, 235471805, 46621242074, 4118765210]
        models.predict_btc_price(input_data, model1, scaler1)
    except:
        pass
if __name__ == "__main__":
    run()


