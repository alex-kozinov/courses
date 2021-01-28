import os
import numpy as np

def check_test(test_name, solutions):
    """Проверка тестов
    Вход: название теста; пользовательское решение
    """
    
    try:
        tests_files = os.listdir(os.path.join('tests', test_name))
    except Exception as e:
        print("Error!")
        print(e)
    
    tests_files = [test_file for test_file in tests_files if '.npy' in test_file]
    assert len(tests_files) == len(solutions)
    
    try:
        for i, test_file in enumerate(sorted(tests_files)):
            answer = np.load(os.path.join('tests', test_name, test_file))
            np.testing.assert_almost_equal(answer, solutions[i])
    except Exception as e:
        print("Wrong answer in test {}!".format(i + 1))
        print(e)  
    else:
        print('OK!')
        