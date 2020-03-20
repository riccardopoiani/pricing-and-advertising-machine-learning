import os
from datetime import datetime


def handle_folder_creation(result_path: str):
    """
    Handle the creation of a folder and return a file in which it is possible to write in that folder, moreover
    it also returns the path to the result path with current datetime appended

    :param result_path: basic folder where to store results
    :return (descriptor to a file opened in write mode within result_path folder. The name of the file
    is result.txt, folder path with the date of the experiment)
    """
    date_string = datetime.now().strftime('%b%d_%H-%M-%S/')
    output_folder_path = result_path + date_string
    output_file_name = output_folder_path + "results.txt"
    try:
        if not os.path.exists(output_folder_path):
            os.mkdir(output_folder_path)
    except FileNotFoundError as e:
        os.makedirs(output_folder_path)

    fd = open(output_file_name, "w")
    return fd, output_folder_path
