import os


def run_python_file(working_directory, file_path, args=None):
    try:
        working_dir_abs = os.path.abspath(working_directory)
        file_path_abs = os.path.normpath(
            os.path.join(working_dir_abs, file_path))
        if os.path.commonpath(
                [working_dir_abs, file_path_abs]) != working_dir_abs:
            return f'Error: Cannot write to "{file_path}" as it is outside the permitted working directory'
        if os.path.isfile(file_path_abs):
            return f'Error: "{file_path}" does not exist or is not a regular file'
    except Exception as e:
        return f'Error running file "{file_path}": {e}"'
