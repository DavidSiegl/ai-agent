from functions.get_file_content import get_file_content


def test():
    result = get_file_content("calculator", "main.py")
    print(result)
    print("")

    result = get_file_content("calculator", "pkg/calculator.py")
    print(result)
    print("")

    result = get_file_content("calculator", "/bin/cat")
    print(result)
    print("")

    result = get_file_content("calculator", "pkg/does_not_exist.py")
    print(result)
    print("")


if __name__ == "__main__":
    test()
