import os

class Dataset:
    @staticmethod
    def get_rgb_list_file(path):
        return os.path.join(path, 'rgb.txt')

    @staticmethod
    def get_depth_list_file(path):
        return os.path.join(path, 'depth.txt')

    @staticmethod
    def get_intrinsics_file(path):
        return os.path.join(path, 'intrinsics.txt')

    @staticmethod
    def get_searcher_file(path):
        return os.path.join(path, 'search_tree.bin')

    @staticmethod
    def get_ground_truth_file(path):
        return os.path.join(path, 'ground_truth.txt')

    @staticmethod
    def get_result_poses_file(path):
        return os.path.join(path, 'all_poses.txt')

    @staticmethod
    def get_known_poses_file(path):
        return os.path.join(path, 'known_poses.txt')

    @staticmethod
    def get_testing_result_file(path):
        return os.path.join(path, 'scores.txt')

    @staticmethod
    def read_lists(filename, value_type=str):
        with open(filename) as file:
            data = file.read()

        lines = data.replace(',', ' ').replace('\t', ' ').splitlines()
        result = []
        for line in lines:
            if len(line) == 0 or line[0] == '#':
                continue
            values = [value_type(value.strip()) for value in line.split() if value.strip() != '']
            result.append(values)

        return result

    @staticmethod
    def read_dict_of_lists(filename, key_type=int):
        lists = Dataset.read_lists(filename)
        file_list = {}
        for values in lists:
            file_list[key_type(values[0])] = values[1] if len(values) == 2 else values[1:]

        return file_list

    @staticmethod
    def write_dict_of_lists(filename, data):
        with open(filename, 'w') as file:
            for key, value in data.items():
                if type(value) == list:
                    value_str = ' '.join([str(item) for item in value])
                else:
                    value_str = str(value)

                file.write('{} {}\n'.format(key, value_str))

    @staticmethod
    def associate(first_dict, second_dict, max_difference):
        """
        Associate two dictionaries of (stamp,data). As the time stamps never match exactly, we aim
        to find the closest match for every input tuple.

        Input:
        first_dict -- first dictionary of (stamp,data) tuples
        second_dict -- second dictionary of (stamp,data) tuples
        max_difference -- search radius for candidate generation

        Output:
        matches -- list of matched tuples ((stamp1,data1),(stamp2,data2))

        """
        first_keys = list(first_dict.keys())
        second_keys = list(second_dict.keys())
        potential_matches = [(abs(a - b), a, b)
            for a in first_keys
            for b in second_keys
            if abs(a - b) < max_difference]
        potential_matches.sort()
        matches = []
        for diff, a, b in potential_matches:
            if a in first_keys and b in second_keys:
                first_keys.remove(a)
                second_keys.remove(b)
                matches.append((a, b))

        matches.sort()
        return matches
