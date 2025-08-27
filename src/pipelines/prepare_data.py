from src.data_prep import load_data

if __name__ == "__main__":
    df = load_data(save_processed=True)
    print("Processed data saved:", df.shape, "-> data/processed/US_House_Price_clean.csv")
