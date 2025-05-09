# sort word in data/common_words.txt
# and save to data/common_words.txt

# Read the words from the file
with open("app/data/common_word.txt", "r") as file:
    words = file.readlines()

# Sort the words
sorted_words = sorted(word.strip() for word in words)

# Write the sorted words back to the file
with open("app/data/common_word.txt", "w") as file:
    file.write("\n".join(sorted_words))
