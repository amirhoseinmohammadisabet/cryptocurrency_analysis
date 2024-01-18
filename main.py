import collector as col

def run():
    data = col.read_data_pd("Data/data.csv")
    print(data)
if __name__ == "__main__":
    run()


