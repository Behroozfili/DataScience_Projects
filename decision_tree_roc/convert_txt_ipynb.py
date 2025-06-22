input_filename = "lab_spamdata_ClassTree.ipynb.txt"
output_filename = "lab_spamdata_ClassTree.ipynb.ipynb"

try:
    with open(input_filename, 'r', encoding='utf-8') as infile:
        content = infile.read()

    with open(output_filename, 'w', encoding='utf-8') as outfile:
        outfile.write(content)

    print(f"The file '{output_filename}' was created successfully.")
    print("You can now open it in Jupyter Notebook or a similar environment.")

except FileNotFoundError:
    print(f"Error: Input file '{input_filename}' not found.")
except Exception as e:
    print(f"An error occurred: {e}")
