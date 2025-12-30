import os


def read_txt(file_path):
    with open(file_path, 'r') as file:
        content = file.read().split('\n\n')
        content = [int(_) for _ in content if _.isdigit()]
        NCCQ = content.count(10)
        NCIQ = content.count(0)
        NPCQ = len(content) - NCCQ - NCIQ
        TSP = sum(content) / (len(content) * 10) if len(content) > 0 else 0
        filtered_list = [element for element in content if element != 0 and element != 10]
        avg_filtered = sum(filtered_list) / len(filtered_list) if len(filtered_list) > 0 else 0

        return {
            'FCR': NCCQ / len(content) if len(content) > 0 else 0,
            'PCR': NPCQ / len(content) if len(content) > 0 else 0,
            'FIR': NCIQ / len(content) if len(content) > 0 else 0,
            'PCRate': avg_filtered/10,
            'ALL': TSP
        }


def process_txt_files(directory_path):
    metrics = ['FCR', 'PCR', 'FIR', 'PCRate', 'ALL']
    print(f"{'File Name':<20} {'FCR':>10} {'PCR':>10} {'FIR':>10} {'PCRate':>10} {'ALL':>10}")

    for file_name in os.listdir(directory_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(directory_path, file_name)
            results = read_txt(file_path)
            print(f"{file_name:<20}", end='')
            for metric in metrics:
                print(f"{results[metric]*100:>10.2f}", end='')
            print()


directory_path = '.'
process_txt_files(directory_path)

