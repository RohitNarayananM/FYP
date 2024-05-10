from datasets import load_dataset, load_from_disk

data = load_dataset("csv", data_files="data.csv")

data = data["train"]

data = data.train_test_split(test_size=0.2)

data.save_to_disk("data")

print(data)
