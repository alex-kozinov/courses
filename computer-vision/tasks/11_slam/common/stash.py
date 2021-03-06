def check_unions(self):
    for root, point_pos in self.union_poses.items():
        self.is_union_valid.setdefault(root, False)
        i, x = root
        kp = self.kp[i][x]
        camera_pos = self.pos[i]
        image_proj = project_position(point_pos, camera_pos, self.intrinsics)
        if True:
            self.is_union_valid[root] = True


def compute_union_poses(self):
    unions_list_of_poses = {}
    for vertex, poses in self.vertex_poses.items():
        root = KPWarehouse.find_set(vertex, self.union_set)
        unions_list_of_poses.setdefault(root, [])
        unions_list_of_poses[root] += poses
    for root, poses in unions_list_of_poses.items():
        poses_np = np.array(poses)
        avg_pos = poses_np.mean(axis=0)
        self.union_poses[root] = avg_pos


def compute_vertex_poses(self):
    for j in range(self.n):
        for i in range(j):
            kp_i = []
            kp_j = []
            pairwise_matches = self.get_pairwise_matches_by_ids(i, j)
            for match in pairwise_matches:
                kp_i.append(self.kp[i][match[0].trainIdx])
                kp_j.append(self.kp[j][match[0].queryIdx])

            kp_coords = triangulation(kp_i, kp_j, self.pos[i], self.pos[j])
            for coord, match in zip(kp_coords, pairwise_matches):
                edge_start = (i, match[0].trainIdx)
                edge_finish = (j, match[0].queryIdx)
                self.vertex_poses.setdefault(edge_start, [])
                self.vertex_poses.setdefault(edge_finish, [])
                self.vertex_poses[edge_start].append(coord)
                self.vertex_poses[edge_finish].append(coord)


def check_edges(self):
    edges = deepcopy(self.edges)
    start_vert = list(map(lambda x: x[0], edges))
    finish_vert = list(map(lambda x: x[1], edges))
    start_vert_unique = set(start_vert)
    finish_vert_unique = set(finish_vert)
    if len(start_vert) != len(start_vert_unique) or len(finish_vert) != len(finish_vert_unique):
        print("Lol")
        assert False


@staticmethod
def find_set(v, union):
    if v == union[v]:
        return v

    union[v] = KPWarehouse.find_set(union[v], union)
    return union[v]


@staticmethod
def union_sets(v, u, union):
    a = KPWarehouse.find_set(v, union)
    b = KPWarehouse.find_set(u, union)
    if a != b:
        union[b] = a
    return union


def compute_union_set(self):
    for vertex in self.vertexes:
        self.union_set[vertex] = vertex

    for edge in self.edges:
        vertex1, vertex2 = edge
        if np.random.randint(2) == 1:
            vertex1, vertex2 = vertex2, vertex1

        self.union_sets(vertex1, vertex2, self.union_set)


def compute_edges(self):
    for j in range(self.n):
        for i in range(j):
            pairwise_matches = self.get_pairwise_matches(i, j)
            for match in pairwise_matches:
                first_vert = (i, match[0].trainIdx)
                second_vert = (j, match[0].queryIdx)
                self.edges.append((first_vert, second_vert))


def compute_vertexes(self):
    self.vertexes = list(set(map(lambda x: x[0], self.edges)) | set(map(lambda x: x[1], self.edges)))

#
# img_i = cv.imread(os.path.join(self.data_dir, self.train_images[i]), 0)
# img_j = cv.imread(os.path.join(self.data_dir, self.train_images[j]), 0)
# curr_img = cv.imread(os.path.join(self.data_dir, test_images[j]), 0)
# matches = self.matcher.get_pairwise_matches_by_ids(i, j)
# matches = [[m] for m in matches]
# matches_1 = [[m] for m in matches_1]
# matches_2 = [[m] for m in matches_2]
# img_ij = cv.drawMatchesKnn(img_i, self.train_kp[i], img_j, self.train_kp[j], matches, None, flags=2)
# img_cur_i = cv.drawMatchesKnn(img_i, kps1, curr_img, test_kp, matches_1, None, flags=2)
# img_cur_j = cv.drawMatchesKnn(img_j, kps2, curr_img, test_kp, matches_2, None, flags=2)
# plt.imshow(img_ij, )
# plt.imshow(img_cur_i, )
# plt.imshow(img_cur_j, )
# plt.show()
# break


# matches_pairs = {}
# for match_group in [matches_1, matches_2]:
#     for match in match_group:
#         x = match.queryIdx
#         matches_pairs.setdefault(x, [])
#         matches_pairs[x].append(match)
# object_points = []
# image_points = []
# for x, match_pair in matches_pairs.items():
#     if len(match_pair) < 2:
#         continue
#     if match_pair[0].trainIdx != match_pair[1].trainIdx:
#         continue
#     object_points.append(points_3d[x])
#     image_points.append(test_kp[match_pair[1].trainIdx].pt)
# if len(object_points) < 8:
#     continue
# object_points = np.array(object_points)
# image_points = np.array(image_points)
# rvec, tvec = compute_camera_pos(object_points, image_points, self.intrinsics)
# matrix = rvec_tvec_to_matric(rvec, tvec)
# res.append(matrix)
# find_answer = True
# break