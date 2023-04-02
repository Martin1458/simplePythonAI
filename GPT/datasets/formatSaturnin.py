import chardet

# Detect the encoding of the file
with open(r"GPT\datasets\saturnin.txt", 'rb') as f:
    encoding = chardet.detect(f.read())['encoding']

# Read the contents of the file
with open(r"GPT\datasets\saturnin.txt", encoding=encoding) as f:
    lines = f.readlines()

# Remove lines that don't start with a tab
lines = [line for line in lines if line.startswith('\t') or line.strip() == '']

# Remove tab from lines that start with a tab
lines = [line.lstrip('\t') for line in lines]

# Save the modified text to the same file
with open(r"GPT\datasets\saturninV2.txt", 'w', encoding=encoding) as f:
    f.write(''.join(lines))