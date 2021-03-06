class Intrinsics:
    fx = 525.0
    fy = 525.0
    cx = 319.5
    cy = 239.5

    def write(self, filename):
        with open(filename, 'w') as file:
            file.write('# fx fy cx cy\n')
            file.write('{} {} {} {}\n'.format(self.fx, self.fy, self.cx, self.cy))

    @staticmethod
    def read(filename):
        with open(filename, 'r') as file:
            data = file.read()
        lines = data.replace(',', ' ').replace('\t', ' ').splitlines()

        for line in lines:
            if len(line) == 0 or line[0] == '#':
                continue

            values = [float(value) for value in line.split()]
            assert len(values) == 4
            intrinsics = Intrinsics()
            intrinsics.fx = values[0]
            intrinsics.fy = values[1]
            intrinsics.cx = values[2]
            intrinsics.cy = values[3]

            return intrinsics

        raise RuntimeError("Wrong intrinsics file format: {}".format(filename))
