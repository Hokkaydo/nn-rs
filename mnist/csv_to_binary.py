import pandas as pd
import numpy as np

def csv_to_binary(input_csv, output_file):
    df = pd.read_csv(input_csv).astype(np.uint8)
    df = df.head(1000)

    with open(output_file, "wb") as f:
        f.write(len(df).to_bytes(4, byteorder="big"))

        for _, row in df.iterrows():
            label = int(row[0])
            pixels = row[1:].to_numpy(dtype=np.uint8)

            f.write(label.to_bytes(1, "big"))
            f.write(pixels.tobytes())

if __name__ == "__main__":
    csv_to_binary("train.csv", "train.bin")
    csv_to_binary("test.csv", "test.bin")
